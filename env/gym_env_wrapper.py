"""
gym_env_wrapper.py - Gym环境包装器

【功能定位】
    作为强化学习训练的核心桥梁，连接三个关键组件：
    1. LeggedRobotEnv: 物理仿真环境（提供传感器数据和执行控制）
    2. BIRLTask: 任务定义（提供观察、动作处理、奖励计算）
    3. Runner/PPO: 训练循环（调用step/reset获取经验）

【核心职责】
    1. 封装环境和任务，提供统一的step/reset接口
    2. 处理噪声注入（领域随机化）
    3. 管理环境重置逻辑
    4. 收集调试数据（可选）

【调用关系】
    ┌─────────────────────────────────────────────────────────────────┐
    │  调用者: rl/utils/runner.py                                    │
    │    runner.collect_rollouts() → wrapper.step() / wrapper.reset()│
    ├─────────────────────────────────────────────────────────────────┤
    │  被调用者:                                                     │
    │    LeggedRobotEnv (env/legged_robot.py)                        │
    │      ├─ step_torques(): 执行物理仿真                           │
    │      ├─ step_states(): 更新状态                               │
    │      └─ reset(): 重置环境                                     │
    │    BIRLTask (env/tasks/birl_task.py)                           │
    │      ├─ action(): 处理网络输出                                 │
    │      ├─ step(): 更新任务状态                                   │
    │      ├─ observation(): 获取观察                               │
    │      ├─ critic_observation(): 获取评论家观察                   │
    │      ├─ reward(): 计算奖励                                   │
    │      ├─ terminate(): 判断终止                                 │
    │      └─ reset(): 重置任务                                     │
    └─────────────────────────────────────────────────────────────────┘

【数据流】
    策略网络输出 → wrapper.step(net_out)
                        ↓
              task.action(net_out) → 关节指令
                        ↓
              env.step_torques(joint_act) → 物理仿真
                        ↓
              task.step() + task.observation()
                        ↓
              task.reward() → 计算奖励
                        ↓
              返回: obs, cri_obs, rew_buf, done, info, eval_rew
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import os

import env
from .legged_robot import LeggedRobotEnv
from .tasks import BaseTask
from isaacgym.torch_utils import *
import collections
import torch


class GymEnvWrapper:
    def __init__(self, env: LeggedRobotEnv, task: BaseTask, residual_task=None,
                 dynamic_params=None, debug: bool = False):
        """
        初始化环境包装器
        
        参数:
            env: LeggedRobotEnv实例，提供物理仿真能力
            task: BaseTask实例（通常是BIRLTask），定义任务逻辑
            residual_task: 残余任务（可选，用于分层强化学习）
            dynamic_params: 动态参数（可选，用于TCN等序列模型）
            debug: 是否启用调试模式（记录详细数据）
        """
        self.env = env                    # 物理仿真环境引用
        self.task = task                  # 任务定义引用
        self.task.debug = debug           # 传递调试标志给任务
        self.device = env.device          # 计算设备（CPU/GPU）
        self.debug = debug                # 调试模式标志
        self.debug_data = {name: [] for name in self.debug_name} if debug else None  # 调试数据缓冲区
        
        # TCN序列模型支持（可选）
        if dynamic_params is not None:
            self.dynamic_params = dynamic_params
            self.tcn_obs_buf = collections.deque(maxlen=dynamic_params['seq_length'])
        
        self.residual_task = residual_task if residual_task is not None else None  # 残余任务（分层RL用）

    def reset(self, env_ids, reset_joint_pos=None, reset_joint_vel=None, reset_base_quat=None):
        """
        重置指定环境
        
        参数:
            env_ids: 需要重置的环境索引列表
            reset_joint_pos: 自定义关节位置（可选）
            reset_joint_vel: 自定义关节速度（可选）
            reset_base_quat: 自定义基座姿态（可选）
            
        返回:
            obs: 演员观察向量
            cri_obs: 评论家观察向量
        """
        # 设置自定义重置参数（用于特定初始条件测试）
        if reset_joint_pos is not None:
            self.env.reset_joint_pos = reset_joint_pos
        if reset_joint_vel is not None:
            self.env.reset_joint_vel = reset_joint_vel
        if reset_base_quat is not None:
            self.env.reset_base_quat = reset_base_quat
        
        # 执行一步仿真以初始化状态
        self.env.step_torques(torch.clone(self.task.current_joint_act))
        self.env.step_states()
        
        # 获取初始观察
        obs = self.task.observation()
        cri_obs = self.task.critic_observation()
        
        # 清空调试数据（如果启用）
        if self.debug:
            self.clear_debug_data()
        
        # 执行环境和任务重置
        self.env.reset(env_ids)
        self.task.reset(env_ids)
        if self.residual_task is not None:
            self.residual_task.reset(env_ids)
        
        return obs, cri_obs

    def step(self, net_out, it=None):
        """
        执行一步仿真（核心方法！）
        
        参数:
            net_out: 策略网络输出张量
            it: 当前迭代次数（可选，用于调试）
            
        返回:
            obs: 演员观察向量
            cri_obs: 评论家观察向量
            rew_buf: 每个环境的总奖励（已求和并裁剪）
            done: 终止标志
            info: 额外信息
            eval_rew: 评估奖励
        
        执行流程:
            1. 处理网络输出 → 生成关节动作指令
            2. 执行多次物理仿真（decimation次，默认3次）
            3. 更新任务状态
            4. 获取观察和奖励
            5. 处理终止和重置
        """
        # ========== 1. 处理网络输出 ==========
        # 将策略网络输出转换为关节动作指令
        joint_act = self.task.action(net_out)
        
        # ========== 2. 执行物理仿真 ==========
        # decimation: 每个策略步执行多次物理仿真（默认3次，每次约3.3ms）
        # 这样策略更新频率为 10ms，物理仿真频率为 3.3ms
        for m in range(self.env.cfg.pd_gains.decimation):
            # 领域随机化：每隔5步重新采样噪声
            if self.env.cfg.noise_values.randomize_noise and m % 5 == 0:
                self.env.lin_vel_noise = self.env.cfg.noise_values.lin_vel * (2. * torch.rand_like(self.env.base_lin_vel) - 1.)
                self.env.gravity_noise = self.env.cfg.noise_values.gravity * (2. * torch.rand_like(self.env.base_euler) - 1.)
                self.env.ang_vel_noise = self.env.cfg.noise_values.ang_vel * (2. * torch.rand_like(self.env.base_ang_vel) - 1.)
                self.env.foot_frc_noise = self.env.cfg.noise_values.foot_frc * (2. * torch.rand_like(self.env.foot_frc) - 1.)
                self.env.joint_pos_noise = self.env.cfg.noise_values.dof_pos * (2. * torch.rand_like(self.env.joint_pos) - 1.)
                self.env.joint_vel_noise = self.env.cfg.noise_values.dof_vel * (2. * torch.rand_like(self.env.joint_vel) - 1.)
                self.env.base_acc_noise = self.env.cfg.noise_values.base_acc * (2. * torch.rand_like(self.env.base_acc) - 1.)
            
            # 执行一次物理仿真（应用关节扭矩）
            self.env.step_torques(joint_act)
        
        # 更新状态（10ms 更新周期）
        self.env.step_states(it)
        
        # ========== 3. 更新任务状态 ==========
        self.task.step()
        
        # ========== 4. 获取观察和奖励 ==========
        obs = self.task.observation()           # 演员观察
        cri_obs = self.task.critic_observation() # 评论家观察
        rew, eval_rew = self.task.reward()       # 奖励（训练用 + 评估用）
        done = self.task.terminate()             # 终止标志
        info = self.task.info()                  # 额外信息
        
        # ⭐ 关键步骤：将奖励数组按行求和，裁剪到非负
        # rew 形状: (num_envs, num_rew_items)
        # rew_buf 形状: (num_envs,)
        rew_buf = torch.clip(rew.sum(dim=1), min=0.)
        
        # ========== 5. 调试数据记录 ==========
        if self.debug and self.env.common_step_counter >= 8:
            self.record_debug_data(rew, obs)
        
        # ========== 6. 处理终止和重置 ==========
        reset_env_ids = (done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
        if len(reset_env_ids) > 0:
            self.env.reset(reset_env_ids)
            self.task.reset(reset_env_ids)
        
        return obs, cri_obs, rew_buf, done, info, eval_rew


    def close(self):
        """关闭环境（预留接口）"""
        pass

    def record_debug_data(self, rew, obs):
        """
        记录调试数据（用于分析训练过程）
        
        参数:
            rew: 原始奖励数组
            obs: 观察向量
            
        记录的数据包括:
            - 奖励、命令、速度、加速度
            - 基座状态（欧拉角、角速度、位置）
            - 脚部状态（位置、速度、接触力、相位）
            - 关节状态（动作、位置、速度、扭矩、加速度）
            - 网络输出
        """
        # 奖励和命令
        self.debug_data['reward'].append(rew / self.env.dt)
        self.debug_data['command'].append(self.task.commands.clone())
        
        # 基座线性和角运动
        self.debug_data['lin_vel'].append(self.task.base_lin_vel.clone())
        self.debug_data['lin_acc'].append(self.task.base_acc.clone())
        self.debug_data['base_eul'].append(self.task.base_euler.clone())
        self.debug_data['ang_vel'].append(self.task.base_ang_vel.clone())
        self.debug_data['base_pos'].append(self.env.base_pos_hd.clone())
        
        # 脚部状态
        self.debug_data['foot_pos'].append(self.env.foot_pos_hd.clone())
        self.debug_data['foot_vel'].append(self.env.foot_vel.clone())
        self.debug_data['foot_frc'].append(self.task.foot_frc.clone())
        self.debug_data['foot_rpy'].append(self.env.foot_euler.clone())
        self.debug_data['foot_phs'].append(self.task.phase_modulator.phase.clone())
        
        # 关节状态
        self.debug_data['joint_act'].append(self.task.current_joint_act.clone())
        self.debug_data['joint_pos'].append(self.task.joint_pos.clone())
        self.debug_data['joint_vel'].append(self.task.joint_vel.clone())
        self.debug_data['joint_tau'].append(self.env.react_tau.clone())
        self.debug_data['joint_acc'].append(self.env.joint_acc.clone())
        
        # 网络输出
        self.debug_data['netout_a'].append(self.task.net_out_history[-1].clone())
        self.debug_data['netout_f'].append(self.task.pm_f.clone())
        self.debug_data['obs_state'].append(obs)

    @property
    def debug_name(self):
        """
        获取调试数据的字段名称（用于Excel输出时的列标题）
        
        返回:
            OrderedDict: 字段名到列标题列表的映射
        """
        d = OrderedDict()
        axises = ['x', 'y', 'z']
        foot_names = ['L', 'R']  # 左脚、右脚
        
        # 奖励和命令
        d['reward'] = self.task.rew_names
        d['command'] = ['fwd_vel', 'lat_vel', 'yaw_rate', 'heading']
        
        # 基座运动
        d['lin_vel'] = [n for n in axises]
        d['lin_acc'] = [n for n in axises]
        d['base_eul'] = [n for n in axises]
        d['ang_vel'] = [n for n in axises]
        d['base_pos'] = [n for n in axises]
        
        # 关节状态
        d['joint_act'] = self.env.dof_names[:10]
        d['joint_pos'] = [n for n in self.env.dof_names[:10]]
        d['joint_vel'] = [n for n in self.env.dof_names]
        d['joint_acc'] = [n for n in self.env.dof_names]
        d['joint_tau'] = [n for n in self.env.dof_names]
        
        # 网络输出
        d['netout_f'] = [f for f in foot_names]
        d['netout_a'] = [f for f in foot_names] + [n for n in self.env.dof_names]
        
        # 脚部状态
        d['foot_phs'] = [n for n in foot_names]
        d['foot_frc'] = [n for n in foot_names]
        d['foot_pos'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['foot_vel'] = [f'{o}_{n}' for o in foot_names for n in axises]
        d['foot_rpy'] = [f'{o}_{n}' for o in foot_names for n in axises]
        
        # 观察状态
        d['obs_state'] = ['obs' + '_' + str(i) for i in range(self.env.num_observations)]
        
        return d


    def clear_debug_data(self):
        """清空调试数据缓冲区"""
        for k, v in self.debug_data.items():
            self.debug_data[k].clear()

    def save_debug_data(self, debug_dir: str):
        """
        将调试数据保存到Excel文件
        
        参数:
            debug_dir: 保存目录路径
            
        返回:
            debug_data: 合并后的调试数据字典
        """
        # 将列表转换为张量，再转为numpy数组
        debug_data = {key: torch.stack(self.debug_data[key], dim=1).cpu().numpy() for key in self.debug_name.keys()}
        
        # 保存前2个环境的数据到Excel
        for i in range(min(self.env.num_envs, 2)):
            data_path = os.path.join(debug_dir, f'debug_{i}.xlsx')
            with pd.ExcelWriter(data_path) as f:
                for key in self.debug_name.keys():
                    pd.DataFrame(np.asarray(debug_data[key][i]), columns=self.debug_name[key]).to_excel(f, key, index=False)
            print(f'#The debug data has been written into `{data_path}`.')
        
        return debug_data