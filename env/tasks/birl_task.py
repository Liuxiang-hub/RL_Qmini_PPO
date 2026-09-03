"""
birl_task.py - BIRL(Bipedal Robot Learning)任务类

功能:
  定义双足机器人强化学习任务,包括:
  1. 观察空间定义
  2. 动作空间定义
  3. 奖励函数设计(核心!)
  4. 命令生成与采样
  5. 相位调制器(步态周期管理)

在机器人走路中的作用:
  - 相当于机器人的"教练",告诉机器人什么是"好行为"
  - 通过奖励函数引导机器人学习稳定走路
  - 提供观察信息给策略网络做决策
  - 将策略输出转换为关节动作

调用关系:
  ┌─────────────────────────────────────────────────────────────────┐
  │                      调用者 (谁调用本类)                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  GymEnvWrapper (env/gym_env_wrapper.py)                        │
  │    ├─ reset(): 调用 task.reset() + task.observation()          │
  │    ├─ step(): 调用 task.action() + task.step() + task.reward() │
  │    └─ 传递网络输出给 task.action()                             │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                     被调用者 (本类调用谁)                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  LeggedRobotEnv (env/legged_robot.py)                          │
  │    ├─ 获取传感器数据: joint_pos, joint_vel, base_euler          │
  │    ├─ 获取接触信息: foot_frc, foot_pos_hd                      │
  │    └─ 获取IMU数据: base_acc, base_lin_vel                      │
  │                                                                 │
  │  PhaseModulator (env/utils/phase_modulator.py)                  │
  │    ├─ phase_modulator.reset(): 重置步态相位                     │
  │    ├─ phase_modulator.compute(): 更新步态频率                   │
  │    └─ phase_modulator.phase: 获取当前相位                       │
  │                                                                 │
  │  BaseTask (env/tasks/base_task.py)                             │
  │    ├─ 继承: _resample_commands(), terminate(), info()          │
  │    └─ 命令生成与终止判断逻辑                                    │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                        类继承关系                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  NullTask → BaseTask → BIRLTask                                │
  │    │            │              │                               │
  │    └─注册机制   └─命令生成     └─双足特定奖励函数               │
  └─────────────────────────────────────────────────────────────────┘

代码结构:
  ┌───────────────────────────────────────────────────────────────┐
  │                       BIRLTask 类结构                         │
  ├───────────────────────────────────────────────────────────────┤
  │                                                               │
  │   __init__(): 初始化                                          │
  │   ├─ 设置命令缓冲区                                           │
  │   ├─ 初始化相位调制器                                          │
  │   ├─ 设置动作边界                                             │
  │   └─ 初始化观察历史缓冲区                                      │
  │                              ↓                                │
  │   reset(): 重置任务                                           │
  │   └─ 重置相位调制器和命令                                      │
  │                              ↓                                │
  │   step(): 每步更新                                            │
  │   ├─ 更新状态延迟                                             │
  │   ├─ 更新相位调制器                                           │
  │   └─ 重采样命令                                               │
  │                              ↓                                │
  │   observation(): 构建观察向量                                  │
  │   └─ 返回历史观察序列                                         │
  │                              ↓                                │
  │   action(): 处理策略输出                                       │
  │   ├─ 缩放动作到合理范围                                       │
  │   └─ 更新相位调制器频率                                        │
  │                              ↓                                │
  │   reward(): 计算奖励(核心!)                                   │
  │   ├─ 平衡奖励                                                 │
  │   ├─ 前进速度奖励                                             │
  │   ├─ 足部接触奖励                                             │
  │   ├─ 动作平滑奖励                                             │
  │   └─ 综合所有奖励                                             │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘

奖励函数设计原则:
  1. 平衡优先: base_heit + balance + twist
  2. 速度跟踪: fwd_vel + yaw_rat + lateral_vel
  3. 步态质量: foot_clr + foot_supt + foot_heit
  4. 动作约束: act_smo + jnt_pos_err + joint_tor

数据流:
  策略网络输出 → action() → 关节指令 → LeggedRobotEnv.step()
                                              ↓
                                    传感器数据更新
                                              ↓
                                    step() → observation()
                                              ↓
                                    策略网络 → (循环)
"""
from math import pi, sin, cos, exp, tau
import numpy as np
from scipy.linalg import toeplitz
from collections import OrderedDict
from env.legged_robot import LeggedRobotEnv
from env.utils.helpers import class_to_dict
from env.utils.math import wrap_to_pi, smallest_signed_angle_between
from env.utils.phase_modulator import PhaseModulator
from env.tasks.null_task import NullTask, register
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R
from env.tasks.base_task import BaseTask
import random
from env.utils.math import scale_transform, smallest_signed_angle_between_torch
from collections import deque
import statistics
import torch


@register
class BIRLTask(BaseTask):
    """
    BIRL(Bipedal Robot Learning)任务类
    
    核心功能:
      1. 定义双足机器人的强化学习任务
      2. 设计奖励函数引导机器人学习走路
      3. 管理步态相位(支撑相/摆动相)
      4. 生成和管理运动命令(速度指令)
      5. 构建观察向量供策略网络使用
    
    关键组件:
      - phase_modulator: 相位调制器,管理步态周期
      - commands: 运动命令缓冲区(x速度, y速度, yaw角速度)
      - obs_history: 观察历史,用于时间序列建模
      - foot_support_mask: 足部支撑状态掩码
    """
    def __init__(self, env: LeggedRobotEnv):
        """ 初始化BIRL任务
        
        Args:
            env (LeggedRobotEnv): 仿真环境实例
        """
        super(BIRLTask, self).__init__(env)
        self.env = env
        self.cmd_id = 0
        self.rew_names = None
        self.num_envs = env.num_envs
        self.num_legs = 2  # 双足机器人

        # 命令缓冲区: [x速度, y速度, yaw角速度, 航向]
        self.commands = torch.zeros(self.num_envs, self.cfg.command.num_commands, dtype=torch.float, device=self.device,
                                    requires_grad=False)

        # 命令配置和重采样间隔
        self.command_cfgs = class_to_dict(self.cfg.command)
        self.resampling_interval = int(self.cfg.command.resampling_time / self.env.dt)
        
        # 静态标志: 判断是否处于静止状态(速度小于0.15视为静止)
        self.static_flag = torch.where(torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15, False,
                                       True).float()
        self.zero_command_env_ids = (torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15).nonzero(as_tuple=False)[:, [0]].flatten()
        
        # 初始化命令
        self._resample_commands(torch.arange(env.num_envs, device=self.device))

        # 观察延迟设置(领域随机化)
        if self.cfg.domain_rand.delay_observation:
            self.delay_joint_steps = random.randint(self.cfg.domain_rand.delay_joint_ranges[0],
                                                    self.cfg.domain_rand.delay_joint_ranges[1])
            self.delay_rate_steps = random.randint(self.cfg.domain_rand.delay_rate_ranges[0],
                                                   self.cfg.domain_rand.delay_rate_ranges[1])
            self.delay_angle_steps = random.randint(self.cfg.domain_rand.delay_angle_ranges[0],
                                                    self.cfg.domain_rand.delay_angle_ranges[1])
        else:
            self.delay_joint_steps = 1
            self.delay_rate_steps = 1
            self.delay_angle_steps = 1
        
        # 相位转换参数(支撑相到摆动相的转换角度)
        self.convert_phi = 1.2 * pi

        # 相位调制器(管理步态周期)
        # 步态相位调制器: 管理双腿的步态周期
        self.phase_modulator = PhaseModulator(time_step=env.dt, num_envs=self.num_envs, num_legs=self.num_legs,device=self.device)
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=torch.arange(self.num_envs),
                                   render=self.env.render or self.env.debug or self.env.epochs > 1 or self.env.tcn_name is not None)
        self.foot_phase = self.phase_modulator.phase  # 当前步态相位
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)  # 相位编码(sin/cos)

        # 动作边界设置
        if self.cfg.action.use_increment:
            self.action_low = to_torch(self.cfg.action.inc_low_ranges, device=self.device)
            self.action_high = to_torch(self.cfg.action.inc_high_ranges, device=self.device)
        else:
            self.action_low = to_torch(self.cfg.action.low_ranges, device=self.device)
            self.action_high = to_torch(self.cfg.action.high_ranges, device=self.device)
            self.action_low[self.num_legs:self.num_legs + self.env.num_dofs] = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device)
            self.action_high[self.num_legs:self.num_legs + self.env.num_dofs] = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device)
        
        # 当前和历史关节动作
        self.current_joint_act = to_torch(self.env.default_dof_pos, device=self.device).repeat(self.num_envs, 1)
        self.previous_joint_act = self.current_joint_act.clone()

        # 参考关节位置(站立姿态)
        self.ref_joint_action = to_torch(self.cfg.action.ref_joint_pos, device=self.device).repeat(self.num_envs, 1)
        
        # 关节动作边界
        self.joint_action_limit_low = torch.as_tensor(self.env.dof_pos_limits[:, 0], device=self.device).repeat(self.num_envs, 1)
        self.joint_action_limit_high = torch.as_tensor(self.env.dof_pos_limits[:, 1], device=self.device).repeat(self.num_envs, 1)

        # 观察和动作历史缓冲区(用于时间序列建模)
        self.obs_history = deque(maxlen=3)
        self.cri_obs_history = deque(maxlen=3)
        self.action_history = deque(maxlen=3)
        self.net_out_history = deque(maxlen=3)

        # 初始化历史缓冲区
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_joint_act)
        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(torch.zeros_like(self.action_low).repeat(self.num_envs, 1))

        # 足部支撑/摆动掩码
        # 步态相位掩码(区分支撑相和摆动相)
        foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
        foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
        self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)  # 支撑相掩码
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)                      # 摆动相掩码
        self.pm_f = self.phase_modulator.frequency.clone()  # 步态频率

        # 足部接触力历史
        self.last_foot_frc = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.foot_frc_acc = torch.zeros(self.num_envs, self.num_legs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_foot_vel = torch.zeros(self.num_envs, self.num_legs * 3, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        # 延迟状态(模拟传感器延迟)
        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.base_acc = self.env.base_acc_his.delay(self.delay_rate_steps)

        # 关节误差和扭矩
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        self.joint_tau = self.env.p_gains * self.joint_pos_error - self.env.d_gains * self.joint_vel
        
        # 足部状态
        self.foot_pos_hd = self.env.foot_pos_hd
        if self.cfg.terrain.mesh_type in ['trimesh','heightfield']:
            self.foot_height = self.env.get_foot_height_to_ground()
        else:
            self.foot_height =  self.env.foot_pos_hd[:, [2, 5]]
        self.foot_vel = self.env.foot_vel
        self.foot_frc = self.env.foot_frc
        
        # 基座状态
        self.base_ang_vel = self.env.base_ang_vel_his.delay(self.delay_rate_steps)
        self.base_euler = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_lin_vel = self.env.base_lin_vel

        # 初始化观察历史
        for _ in range(self.obs_history.maxlen):
            self.obs_history.append(self.pure_observation())
        for _ in range(self.cri_obs_history.maxlen):
            self.cri_obs_history.append(self.pure_critic_observation())

        # 额外信息(用于记录)
        self.extra_info["task"] = {}
        if self.cfg.terrain.curriculum:
            self.extra_info["task"]["terrain_level"] = torch.mean(self.env.terrain_levels.float())
        if self.cfg.runner.send_timeouts:
            self.extra_info["timeouts"] = self.env.time_out_buf

    def reset(self, env_ids):
        """ 重置指定环境的任务状态
            
        Args:
            env_ids (list[int]): 需要重置的环境ID列表
        """
        # 重置延迟状态
        self.base_acc[env_ids] = self.env.base_acc_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_vel[env_ids] = self.env.joint_vel_his.delay(self.delay_joint_steps)[env_ids]
        self.joint_pos[env_ids] = self.env.joint_pos_his.delay(self.delay_joint_steps)[env_ids]
        self.base_ang_vel[env_ids] = self.env.base_ang_vel_his.delay(
            self.delay_rate_steps
        )[env_ids]
        self.base_euler[env_ids] = self.env.base_eul_his.delay(
            self.delay_angle_steps
        )[env_ids]
        self.base_lin_vel[env_ids] = self.env.base_lin_vel[env_ids]
        
        # 重置关节动作
        self.current_joint_act[env_ids] = self.env.default_dof_pos
        self.previous_joint_act[env_ids] = self.current_joint_act[env_ids].clone()

        # 更新关节误差
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        
        # 重置相位调制器
        self.phase_modulator.reset(convert_phi=self.convert_phi, env_ids=env_ids,
                                   render=self.env.render or self.env.epochs > 1 or self.env.tcn_name is not None)
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)
        
        # 更新静态标志
        self.static_flag = torch.where(torch.norm(self.commands[:, :3], dim=1, keepdim=True) < 0.15, False,True).float()
        
        # 更新地形课程学习(如果启用)
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        # 重采样命令
        self._resample_commands(env_ids)
        
        # 更新足部支撑/摆动掩码
        foot_support_mask_1 = torch.where(self.foot_phase >= 0, True, False)
        foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
        self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone()

        # A terminal observation may already contain NaN/Inf. Replace every
        # history frame for reset environments so stale terminal values cannot
        # leak into the next policy input.
        # The simulator state has already been reset, but derived rigid-body
        # tensors are refreshed on the next physics step. A failed PhysX actor
        # can therefore still expose NaN/Inf here for one frame. Use a neutral
        # finite reset frame; the next step replaces it with measured state.
        fresh_obs = torch.nan_to_num(
            self.pure_observation(), nan=0.0, posinf=3.0, neginf=-3.0
        )
        for history_frame in self.obs_history:
            history_frame[env_ids] = fresh_obs[env_ids]
        fresh_cri_obs = torch.nan_to_num(
            self.pure_critic_observation(), nan=0.0, posinf=10.0, neginf=-10.0
        ).clip(min=-10.0, max=10.0)
        for history_frame in self.cri_obs_history:
            history_frame[env_ids] = fresh_cri_obs[env_ids]


    def step(self):
        """ 每步更新任务状态
            
            更新延迟状态、相位调制器、命令重采样等
        """
        # 更新延迟状态
        self.joint_pos = self.env.joint_pos_his.delay(self.delay_joint_steps)
        self.joint_vel = self.env.joint_vel_his.delay(self.delay_joint_steps)
        self.base_acc = self.env.base_acc_his.delay(self.delay_rate_steps).clip(min=-30., max=30.)
        self.joint_tau = self.env.joint_tau_his.delay(1)

        # 更新关节误差
        self.joint_pos_error = self.current_joint_act - self.joint_pos
        
        # 更新足部状态
        self.foot_pos_hd = self.env.foot_pos_hd
        if self.cfg.terrain.mesh_type in ['trimesh', 'heightfield']:
            self.foot_height = self.env.get_foot_height_to_ground()
        else:
            self.foot_height = self.env.foot_pos_hd[:, [2, 5]]

        self.foot_vel = self.env.foot_vel
        self.foot_frc = self.env.foot_frc

        # 更新基座状态
        self.base_euler = self.env.base_eul_his.delay(self.delay_angle_steps)
        self.base_ang_vel = self.env.base_ang_vel_his.delay(self.delay_rate_steps)
        self.base_lin_vel = self.env.base_lin_vel
        
        # 更新相位信息
        self.foot_phase = self.phase_modulator.phase
        self.pm_phase = torch.cat((torch.sin(self.foot_phase), torch.cos(self.foot_phase)), 1)

        # 更新足部支撑/摆动掩码
        foot_support_mask_1 = torch.where(self.foot_phase >= 0., True, False)
        foot_support_mask_2 = torch.where(self.foot_phase < self.convert_phi, True, False)
        self.foot_support_mask = torch.logical_and(foot_support_mask_1, foot_support_mask_2)
        self.foot_swing_mask = torch.logical_not(self.foot_support_mask)
        self.pm_f = self.phase_modulator.frequency.clone().detach()
        
        # 按间隔重采样命令
        env_ids = ((self.env.episode_length_buf) % self.resampling_interval == 0).nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

        # 随机化观察延迟(领域随机化)
        if self.cfg.domain_rand.delay_observation and self.env.common_step_counter % 200 == 0:
            self.delay_joint_steps = random.randint(self.cfg.domain_rand.delay_joint_ranges[0],
                                                    self.cfg.domain_rand.delay_joint_ranges[1])
            self.delay_rate_steps = random.randint(self.cfg.domain_rand.delay_rate_ranges[0],
                                                   self.cfg.domain_rand.delay_rate_ranges[1])
            self.delay_angle_steps = random.randint(self.cfg.domain_rand.delay_angle_ranges[0],
                                                    self.cfg.domain_rand.delay_angle_ranges[1])

        # 单帧维度 = 2(命令) + 2(角度) + 3(角速度) + 10(关节位置偏差) + 10(关节速度) + 10(关节误差) + 4(相位) + 2(频率) = 43维
        #历史拼接后维度 = 43 × 3 = 129维

    def observation(self):
        """ 构建观察向量
            
            返回包含历史观察的序列(用于时间序列模型如TCN/RNN)
            
        Returns:
            torch.Tensor: 观察向量 [num_envs, observation_dim * history_length]
        """
        self.obs_buf_pure = self.pure_observation()
        self.obs_history.append(self.obs_buf_pure)
        return  torch.cat([obs for obs in self.obs_history], dim=-1)

# 单帧观察向量 (43维)
    """ pure_obs = [0.5, 0.0,                              # 命令
            0.05, 0.1,                              # 角度
            0.05, 0.1, 0.0,                         # 角速度
            -0.02, 0.01, -0.03, 0.02, -0.01, 0.02, -0.01, 0.03, -0.02, 0.01,  # 关节位置偏差
            0.1, -0.05, 0.08, -0.03, 0.05, -0.1, 0.05, -0.08, 0.03, -0.05,    # 关节速度
            0.01, -0.02, 0.01, -0.01, 0.02, -0.01, 0.02, -0.01, 0.01, -0.02,  # 关节误差
            0.866, 0.5, -0.866, 0.5,                # 相位
            -0.55, -0.55]                           # 频率

# 拼接3帧历史后 (129维)
# obs = pure_obs_prev2 + pure_obs_prev1 + pure_obs_current
# """
    def critic_observation(self):
        """ 构建Critic网络的观察向量
            
            Critic观察通常比Actor观察更丰富
            
        Returns:
            torch.Tensor: Critic观察向量
        """
        pure_obs_buf = self.pure_critic_observation()
        self.cri_obs_history.append(pure_obs_buf)
        return  torch.cat([obs for obs in self.cri_obs_history], dim=-1)

    def pure_critic_observation(self):
        """ 构建纯Critic观察向量(不包含历史)
            
        Returns:
            torch.Tensor: 纯观察向量
        """
        obs_buf = torch.cat([
            self.commands[:, [0,2]],                              # 目标速度(x, yaw)
            self.commands[:, [0]] - self.env.base_lin_vel[:, [0]], # 速度误差
            self.commands[:, [2]] - self.env.base_ang_vel[:, [2]], # yaw角速度误差
            self.env.base_lin_vel,                                # 基座线速度
            self.env.base_euler[:, :2],                            # 基座欧拉角(roll, pitch)
            self.env.base_ang_vel * 0.5,                          # 基座角速度
            (self.env.joint_pos - self.ref_joint_action),          # 关节位置误差
            self.env.joint_vel * 0.1,                             # 关节速度
            (self.current_joint_act - self.ref_joint_action),      # 动作与参考的误差
            self.joint_pos_error,                                  # 关节位置误差
            self.pm_phase * self.static_flag,                     # 相位信息(运动时)
            (self.pm_f * 0.3 - 1.) * self.static_flag,           # 相位频率(运动时)
            self.net_out_history[-1][:, self.num_legs:] / 15.,     # 网络输出历史
            self.foot_height.clip(min=-0.5, max=0.5) * 10.,       # 足部高度
            (self.env.base_pos_hd[:, [2]] - 0.4) * 10.,            # 基座高度
            self.env.foot_vel.clip(min=-8., max=8.) * 0.5,         # 足部速度
            self.env.base_acc.clip(min=-20., max=20.) * 0.2,       # 基座加速度
            self.env.foot_frc.clip(min=0., max=200.) * 0.01,       # 足部接触力
            self.net_out_history[-1][:, self.num_legs:] / 15.,     # 网络输出历史
            self.base_euler[:, :2] * 1.,                           # 基座欧拉角
            self.base_ang_vel * 0.5,                              # 基座角速度
            self.joint_pos - self.ref_joint_action,                # 关节位置误差
            self.joint_vel * 0.1,                                 # 关节速度
            self.joint_pos_error,                                  # 关节位置误差
        ], dim=1)
        # Bound rare but finite PhysX spikes before they overflow the value
        # network. Nominal scaled critic features remain well inside this range.
        return obs_buf.clip(min=-10.0, max=10.0)

    def pure_observation(self):
        """ 构建纯Actor观察向量(不包含历史)
            
            Actor观察更简洁,专注于关键状态
            
        Returns:
            torch.Tensor: 纯观察向量
        """
        self.obs_buf = torch.cat([
            self.commands[:, [0,2]],                              # 目标速度(x, yaw)
            self.base_euler[:, :2] * 1.,                           # 基座欧拉角(roll, pitch)
            self.base_ang_vel * 0.5,                              # 基座角速度
            self.joint_pos - self.ref_joint_action,                # 关节位置误差
            self.joint_vel * 0.1,                                 # 关节速度
            self.joint_pos_error,                                  # 关节位置误差
            self.pm_phase * self.static_flag,                     # 相位信息(运动时)
            (self.pm_f * 0.3 - 1.) * self.static_flag,           # 相位频率(运动时)
        ], dim=1).clip(min=-3., max=3.)
        return self.obs_buf

    def action(self, net_out):
        """ 处理策略网络输出,转换为关节动作
            
        Args:
            net_out (torch.Tensor): 策略网络输出
            
        Returns:
            torch.Tensor: 关节动作(目标角度)
        """
        # 缩放网络输出到动作范围
        net_out = scale_transform(net_out, self.action_low, self.action_high)
        self.net_out_history.append(net_out)
        
        # 更新相位调制器频率(前num_legs维是频率控制)
        self.phase_modulator.compute(net_out[:, :self.num_legs])
        
        # 更新关节动作
        if self.cfg.action.use_increment:
            # 增量控制:动作叠加到当前关节角度
            self.current_joint_act += net_out[:, self.num_legs:] * self.env.dt
        else:
            # 绝对控制:直接设置关节角度
            self.current_joint_act = net_out[:, self.num_legs:]
        
        # 裁剪到关节角度限制
        self.current_joint_act = torch.clip(self.current_joint_act, self.joint_action_limit_low,self.joint_action_limit_high)
        
        # 更新动作历史
        self.action_history.append(self.current_joint_act.clone())
        self.previous_joint_act = self.current_joint_act.clone()
        
        return self.current_joint_act

    def reward(self):
        """
        计算奖励函数(核心方法!)
        
        奖励函数设计原则:
        1. 平衡奖励: 鼓励机器人保持直立姿态
        2. 速度跟踪奖励: 鼓励机器人跟踪期望速度
        3. 步态质量奖励: 鼓励正确的步态相位和脚部运动
        4. 动作约束奖励: 鼓励平滑、高效的动作
        5. 能量奖励: 惩罚过大的关节扭矩
        
        返回:
            rewards: 所有奖励项的拼接张量(用于训练)
            eval_rew: 评估奖励(用于性能评估)
        """
        # ========== 基础常量 ==========
        constant_rew = to_torch([1.]).repeat(self.num_envs, 1)  # 恒定奖励基线
        lin_vel_x_norm = torch.clip(torch.abs(self.commands[:, [0]]), min=0.3, max=2.) + 0.2  # 归一化系数
        yaw_rate_norm = torch.clip(torch.abs(self.commands[:, [2]]), min=0.3, max=1.5) + 0.2  # yaw速率归一化
        
        # ========== 平衡相关奖励 ==========
        # 基座高度奖励: 鼓励保持目标高度
        posture_cfg = self.cfg.posture_reward
        height_error = self.env.base_pos[:, [2]] - posture_cfg.target_height
        base_heit_rew = torch.exp(-posture_cfg.height_gain * height_error ** 2)

        # 姿态约束提供三种可复现实验模式，便于论文消融对比。
        posture_mode = posture_cfg.mode
        if posture_mode in ('baseline', 'dynamic_reference'):
            posture_reference = torch.zeros_like(self.env.base_euler[:, :2])
            if posture_mode == 'dynamic_reference':
                posture_reference[:, [0]] = torch.clamp(
                    -posture_cfg.lateral_velocity_gain * self.env.base_lin_vel[:, [1]]
                    -posture_cfg.roll_rate_gain * self.env.base_ang_vel[:, [0]],
                    min=-posture_cfg.max_roll_reference,
                    max=posture_cfg.max_roll_reference,
                )
                forward_velocity_error = (
                    self.env.base_lin_vel[:, [0]] - self.commands[:, [0]]
                )
                posture_reference[:, [1]] = torch.clamp(
                    -posture_cfg.forward_velocity_error_gain * forward_velocity_error
                    -posture_cfg.pitch_rate_gain * self.env.base_ang_vel[:, [1]],
                    min=-posture_cfg.max_pitch_reference,
                    max=posture_cfg.max_pitch_reference,
                )

            # 原始 RoboTamer 公式：保留 +1 带来的 0.5 奖励地板。
            balance_rew = 0.5 * (
                base_heit_rew * torch.exp(
                    -torch.clip(5. / lin_vel_x_norm, min=2., max=8.)
                    * torch.norm(
                        self.env.base_euler[:, :2] - posture_reference,
                        dim=-1,
                        keepdim=True,
                    )
                ) + 1.
            )
        elif posture_mode in ('angle_only', 'posture_rate'):
            posture_cost = posture_cfg.angle_gain * torch.sum(
                self.env.base_euler[:, :2] ** 2, dim=1, keepdim=True
            )
            if posture_mode == 'posture_rate':
                posture_cost += posture_cfg.rate_gain * torch.sum(
                    self.env.base_ang_vel[:, :2] ** 2, dim=1, keepdim=True
                )

            # 无奖励地板：高度、姿态或角速度误差过大时奖励可趋近于 0。
            balance_rew = base_heit_rew * torch.exp(-posture_cost)
        else:
            raise ValueError('Unknown posture reward mode: {}'.format(posture_mode))

        # ========== 速度跟踪奖励 ==========
        # 前向速度跟踪: 鼓励跟踪期望前向速度
        forward_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * (
                self.commands[:, [0]] - self.env.base_lin_vel[:, [0]]) ** 2)
        
        # 侧向速度惩罚: 惩罚侧向运动
        lateral_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=3., max=15.) * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True) ** 2)

        # yaw角速度跟踪: 鼓励跟踪期望转向速度
        yaw_rate_rew = torch.exp(-torch.clip(2. / yaw_rate_norm, min=2., max=6.) * (self.commands[:, [2]] - self.env.base_ang_vel[:, [2]]) ** 2)

        # 额外侧向速度惩罚(动态时)
        lateral_vel_rew += -0.6 / lin_vel_x_norm * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True) * self.static_flag

        # 俯仰/滚转角速度惩罚
        ang_vel_rew = torch.exp(
            -torch.clip(2. / lin_vel_x_norm, min=0.7, max=6.) * torch.norm(self.env.base_ang_vel[:, :2], dim=1,
                                                                            keepdim=True) ** 2)
        
        # 基座加速度惩罚(偏离重力方向)
        base_acc_rew = -0.4 / lin_vel_x_norm * torch.norm((self.env.base_acc - to_torch([0, 0, 9.81], device=self.device)) * 0.1, dim=1, keepdim=True)
        base_acc_rew *= self.static_flag

        # 垂直速度惩罚
        vertical_vel_rew = torch.exp(-torch.clip(5. / lin_vel_x_norm, min=2., max=10.) * torch.norm(self.env.base_lin_vel[:, [2]], dim=1,
                                                                           keepdim=True) ** 2)
        vertical_vel_rew -= 0.2 / lin_vel_x_norm * torch.norm(self.env.base_lin_vel[:, 1:], dim=1, keepdim=True) * self.static_flag

        # ========== 步态质量奖励 ==========
        # 检测脚部接触状态
        support_foot_index = torch.where(self.env.foot_frc >= 10., True, False)  # 支撑脚(接触力>=10N)
        swing_foot_index = torch.where(self.env.foot_frc < 1., True, False)     # 摆动脚(接触力<1N)

        # 脚部清除奖励: 鼓励摆动相时脚离开地面
        foot_clear_rew = torch.sum(torch.logical_and(swing_foot_index, self.foot_swing_mask), dtype=torch.float, dim=1, keepdim=True) / self.num_legs

        # 脚部支撑奖励: 鼓励支撑相时脚接触地面
        foot_support_rew = torch.sum(torch.logical_and(support_foot_index, self.foot_support_mask), dtype=torch.float, dim=1, keepdim=True) / self.num_legs
        foot_support_rew *= self.static_flag
        foot_clear_rew *= self.static_flag

        # 脚部高度奖励: 鼓励摆动脚抬起到合适高度(约0.05m)
        foot_heit_score = 40. * torch.clip(self.foot_height, min=0.0, max=0.05)
        foot_height_rew = torch.sum(self.foot_swing_mask * foot_heit_score, dim=1, keepdim=True).clip(max=2.) * self.static_flag

        # 惩罚脚抬得过高
        foot_height_rew += -20. * torch.sum((self.foot_height - 0.06).clip(min=0.), dim=1, keepdim=True)
        # 惩罚支撑脚离地
        foot_height_rew += -0.2 * torch.sum(self.foot_support_mask * foot_heit_score, dim=1, keepdim=True) * self.static_flag
        foot_height_rew += -0.2 * torch.sum(support_foot_index * foot_heit_score, dim=1, keepdim=True) * self.static_flag

        # 扭转惩罚: 惩罚基座倾斜
        twist_rew = -torch.norm(self.env.base_euler[:, :2], dim=-1, keepdim=True)

        # 脚部力变化惩罚(冲击惩罚)
        self.foot_frc_acc = (self.env.foot_frc - self.last_foot_frc).clone()
        foot_soft_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.5) * torch.norm(self.foot_frc_acc, dim=1, keepdim=True) / 100.

        self.last_foot_frc = self.env.foot_frc.clone().detach()

        # 脚部接触力惩罚: 摆动相不应有接触力,支撑相接触力应适中(约55N)
        feet_contact_frc_rew = -torch.norm(self.env.foot_frc * self.foot_swing_mask, dim=1, keepdim=True) * self.static_flag
        feet_contact_frc_rew += -torch.norm((torch.abs(self.env.foot_frc - 55.) * support_foot_index).clip(min=0.), dim=1, keepdim=True)

        # ========== 脚部滑动和速度奖励 ==========
        clip_foot_h = torch.abs(self.foot_height) + 0.03  # 防止除零

        # 脚部滑动奖励: 鼓励摆动脚向前摆动
        foot_slip_rew = 2. * (lin_vel_x_norm * torch.sum(
            (self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, 0]) * self.commands[:, [0]].sign() * self.foot_swing_mask,
            dim=1, keepdim=True)).clip(min=-0., max=1.) * self.static_flag

        # 惩罚侧向滑动
        foot_slip_rew += -0.5 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [1]], dim=-1), dim=1,
                                           keepdim=True) * self.static_flag

        # 静态时允许脚部微动
        foot_slip_rew += 0.3 * torch.norm(torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1), dim=1, keepdim=True) * (
                self.static_flag - 1.)

        # 惩罚支撑脚滑动
        foot_slip_rew += -0.3 / lin_vel_x_norm * torch.norm(
            0.1 * torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, :2], dim=-1) / clip_foot_h * self.foot_support_mask, dim=1,
            keepdim=True) * self.static_flag

        # 脚部垂直速度惩罚(落地冲击)
        foot_vz_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.), dim=-1) / clip_foot_h,
            dim=1, keepdim=True) * self.static_flag

        # 静态时允许脚部微动
        foot_vz_rew += 0.8 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            torch.norm(self.env.foot_vel.view(self.num_envs, self.num_legs, -1)[:, :, [2]].clip(max=0.), dim=-1),
            dim=1, keepdim=True) * (self.static_flag - 1.)

        # 脚部加速度惩罚
        foot_acc_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(self.env.foot_vel[:, [2, 5]], dim=1, keepdim=True)

        # ========== 动作平滑和约束奖励 ==========
        # 动作平滑奖励(二阶差分惩罚): 鼓励平滑的动作变化
        action_smooth_rew = -0.3 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(
            self.action_history[-3] - 2. * self.action_history[-2] + self.action_history[-1], dim=1, keepdim=True)
        
        # 网络输出平滑奖励
        net_out_smooth_rew = -0.2 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.norm(
            (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, self.num_legs:], dim=1, keepdim=True) ** 2

        # 动作约束奖励: 惩罚偏离参考位置的动作
        action_constraint_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, 0, 1.) * torch.norm((self.current_joint_act - self.ref_joint_action), dim=1, keepdim=True)
        action_constraint_rew += -3. * torch.norm(((self.current_joint_act - self.ref_joint_action)[:, [0, 1, 5, 6]]), dim=1, keepdim=True) * self.static_flag

        # 支撑腿约束奖励: 惩罚支撑腿偏离参考位置
        sa_constraint_rew = -0.1 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.current_joint_act - self.ref_joint_action, dim=1, keepdim=True) ** 2 * self.static_flag

        sa_constraint_rew += -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, :5] * support_foot_index[:, [0]]), dim=1,
            keepdim=True) ** 2
        sa_constraint_rew += -self.static_flag * torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(
            ((self.env.joint_pos - self.ref_joint_action)[:, 5:] * support_foot_index[:, [1]]), dim=1,
            keepdim=True) ** 2

        # 关节位置误差惩罚: 惩罚动作与实际关节位置的偏差
        joint_pos_error_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm((self.current_joint_act - self.env.joint_pos), dim=1, keepdim=True) ** 2

        # 关节速度惩罚: 惩罚过大的关节速度
        joint_velocity_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.env.joint_vel[:, :], dim=1, keepdim=True) ** 2
        joint_velocity_rew += -torch.clip(1. / lin_vel_x_norm, 0, 1) * torch.norm(self.env.joint_vel[:, [0, 1, 5, 6]], dim=1, keepdim=True) ** 2

        # 关节扭矩惩罚: 惩罚超过扭矩限制的行为
        joint_tor_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=2.) * torch.sum(
            (torch.abs(self.env.react_tau[:, :]) - self.env.torque_limits[:]).clip(min=0.), dim=1, keepdim=True)

        joint_tor_rew *= self.static_flag

        # ========== 步态频率和其他奖励 ==========
        self.last_foot_vel = self.env.foot_vel.clone().detach()
        
        # 步态频率平滑奖励: 惩罚步态频率的突变
        pmf_rew = -0.02 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(
            (self.net_out_history[-3] - 2 * self.net_out_history[-2] + self.net_out_history[-1])[:, :self.num_legs],
            dim=1, keepdim=True)
        # 惩罚支撑相时调整步态频率
        pmf_rew += -0.5 * torch.clip(1 / lin_vel_x_norm, 0, 1.) * torch.norm(self.net_out_history[-1][:, :self.num_legs] * self.foot_support_mask, dim=1, keepdim=True) ** 2
        pmf_rew *= self.static_flag

        # 网络输出幅值惩罚: 惩罚过大的动作输出
        net_out_val_rew = -0.4 * torch.clip(1. / lin_vel_x_norm, min=0., max=1.) * torch.norm(self.net_out_history[-1][:, self.num_legs:], dim=1, keepdim=True) ** 2
        
        # 脚部俯仰角度惩罚: 惩罚脚部过度倾斜
        foot_py_rew = -0.5 * (torch.norm(self.env.foot_euler[:, [1, 4]], dim=1, keepdim=True))

        # 腿宽奖励: 鼓励保持约0.14m的腿宽
        leg_width_rew = -torch.norm(torch.abs(self.env.foot_pos_hd[:, [1, 4]] - self.env.base_pos_hd[:, [1]]) - 0.14, dim=1, keepdim=True)

        # 步态相位协调奖励: 鼓励双腿交替运动(相位差约180度)
        lsin = torch.sin(self.foot_phase.clone())
        lcos = torch.cos(self.foot_phase.clone())
        foot_phase_rew = -torch.norm(lsin[:, [0]] + lsin[:, [1]], dim=1, keepdim=True) ** 2  # sin(phi1) + sin(phi2) ≈ 0
        foot_phase_rew += -torch.norm(lcos[:, [0]] + lcos[:, [1]], dim=1, keepdim=True) ** 2  # cos(phi1) + cos(phi2) ≈ 0
        foot_phase_rew *= self.static_flag

        # ========== 奖励汇总 ==========
        rew_dict = dict(
            constant=constant_rew * 0.3,           # 恒定基线奖励
            base_heit=base_heit_rew,               # 基座高度奖励
            balance=balance_rew * 1.5,             # 平衡奖励(权重最高)
            fwd_vel=forward_vel_rew * 2.3,         # 前向速度跟踪奖励
            yaw_rat=yaw_rate_rew * 2.5,            # yaw角速度跟踪奖励
            lateral_vel=lateral_vel_rew * 0.7,     # 侧向速度惩罚
            vertical_vel=vertical_vel_rew * 0.6,   # 垂直速度惩罚
            ang_vel=ang_vel_rew * 0.6,             # 角速度惩罚
            twist=twist_rew * 2.5,                 # 扭转惩罚
            base_acc=base_acc_rew * balance_rew * 0.1,           # 基座加速度惩罚
            foot_clr=foot_clear_rew * 1.,                        # 脚部清除奖励
            foot_supt=foot_support_rew * 0.7,                    # 脚部支撑奖励
            foot_heit=foot_height_rew * 0.7,                     # 脚部高度奖励
            leg_width_rew=leg_width_rew * balance_rew * 0.5,     # 腿宽奖励
            act_const=action_constraint_rew * balance_rew * 0.2, # 动作约束惩罚
            sa_const=sa_constraint_rew * balance_rew * 0.1,      # 支撑腿约束惩罚
            foot_phase=foot_phase_rew * balance_rew * 0.3,       # 步态相位协调奖励
            jnt_pos_err=joint_pos_error_rew * balance_rew * 0.2, # 关节位置误差惩罚
            act_smo=action_smooth_rew * balance_rew * 1.5,       # 动作平滑奖励
            net_smo=net_out_smooth_rew * balance_rew * 0.001,    # 网络输出平滑奖励
            net_out_val=net_out_val_rew * balance_rew * 0.0001,  # 网络输出幅值惩罚
            foot_slip=foot_slip_rew * balance_rew * 0.5,         # 脚部滑动奖励
            foot_vz=foot_vz_rew * 0.2 * balance_rew,             # 脚部垂直速度惩罚
            foot_acc=foot_acc_rew * balance_rew * 0.05,          # 脚部加速度惩罚
            foot_sft=foot_soft_rew * 2.7 * balance_rew,          # 脚部力变化惩罚
            jnt_vel=joint_velocity_rew * balance_rew * 0.003,    # 关节速度惩罚
            feet_py=foot_py_rew * balance_rew * 0.5,             # 脚部俯仰惩罚
            feet_frc=feet_contact_frc_rew * 0.001,               # 脚部接触力惩罚
            joint_tor=joint_tor_rew * 0.001,                     # 关节扭矩惩罚
            pmf=pmf_rew * balance_rew * 0.03                     # 步态频率平滑奖励
        )
        
        # 记录奖励项名称(仅首次调试时)
        if self.debug:
            self.rew_names = [name for name in rew_dict.keys()]
            self.debug = None
        
        # 拼接所有奖励项,裁剪到[-4, 5]范围并乘以时间步长
        rewards = torch.cat(
            [torch.clip(value.to(self.device), min=-4., max=5.) * self.env.dt for value in rew_dict.values()], dim=1)
        
        # 评估奖励: 仅使用关键性能指标
        eval_rew = torch.cat([rew_dict[key] * self.env.dt for key in
                              ['fwd_vel', 'yaw_rat', 'ang_vel', 'lateral_vel', 'vertical_vel', 'twist']],
                             dim=1).sum(dim=1)
        
        return rewards, eval_rew


"""
=============================================================================
                        reward() 函数详细说明
=============================================================================

【函数作用】
    计算强化学习奖励函数，是机器人训练的核心驱动力。通过综合30+个奖励项，引导
    机器人学习稳定、高效的双足行走策略。

【输入数据】(通过 self.env 和 self 访问)
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 传感器数据                                                          │
    │   base_euler:     (num_envs, 3)  基座欧拉角 [roll, pitch, yaw]      │
    │   base_lin_vel:   (num_envs, 3)  基座线速度                          │
    │   base_ang_vel:   (num_envs, 3)  基座角速度                          │
    │   base_acc:       (num_envs, 3)  基座加速度                          │
    │   joint_pos:      (num_envs, dofs) 关节位置                          │
    │   joint_vel:      (num_envs, dofs) 关节速度                          │
    │   foot_frc:       (num_envs, 2)   脚部接触力                          │
    │   foot_pos_hd:    (num_envs, 6)   脚部位置(x,y,z)×2                  │
    │   foot_vel:       (num_envs, 6)   脚部速度                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 控制命令                                                            │
    │   commands:       (num_envs, 4)   [x速度, y速度, yaw角速度, 朝向]     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 动作和状态历史                                                        │
    │   current_joint_act:  当前关节动作指令                                │
    │   action_history:     动作历史队列(最近3步)                           │
    │   net_out_history:    网络输出历史队列                                │
    │   phase_modulator:    步态相位调制器                                  │
    └─────────────────────────────────────────────────────────────────────┘

【输出数据】
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 返回值1: rewards                                                    │
    │   形状: (num_envs, num_rew_items)                                   │
    │   用途: 用于策略训练，包含所有26个奖励项                              │
    │   处理: 每行求和 → clip(min=0) → 用于计算优势估计                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 返回值2: eval_rew                                                   │
    │   形状: (num_envs,)                                                 │
    │   用途: 用于性能评估，仅包含6个关键指标                               │
    │   指标: fwd_vel + yaw_rat + ang_vel + lateral_vel + vertical_vel + twist
    └─────────────────────────────────────────────────────────────────────┘

【奖励项分类汇总】
    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ 类别               │ 奖励项名称                                    │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │ 平衡相关(5项)       │ constant, base_heit, balance, twist, ang_vel │
    │ 速度跟踪(3项)       │ fwd_vel, yaw_rat, lateral_vel, vertical_vel  │
    │ 步态质量(7项)       │ foot_clr, foot_supt, foot_heit, foot_slip,   │
    │                     │ foot_vz, foot_acc, foot_sft, feet_py          │
    │ 动作约束(6项)       │ act_const, sa_const, jnt_pos_err, act_smo,   │
    │                     │ net_smo, net_out_val                         │
    │ 能量效率(3项)       │ jnt_vel, feet_frc, joint_tor                 │
    │ 步态协调(2项)       │ leg_width_rew, foot_phase, pmf               │
    └─────────────────────┴──────────────────────────────────────────────┘

【关键设计原则】
    1. 平衡优先: balance_rew 作为乘法因子几乎乘遍所有奖励项
    2. 速度自适应: 惩罚系数与速度成反比(低速更严格)
    3. 平滑约束: 通过二阶差分惩罚鼓励平滑动作
    4. 安全边界: clip(min=-4, max=5)防止奖励爆炸
    5. 时间归一化: 乘以 dt 确保奖励与时间步长无关

【完整示例1: 完美直立行走】
    假设有2个并行环境(num_envs=2):
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 环境0: 完美表现                                                     │
    │   base_euler = [0.0, 0.0, 0.1]    # 几乎直立                        │
    │   base_lin_vel = [0.5, 0.0, 0.0]  # 速度完美匹配命令                 │
    │   commands = [0.5, 0.0, 0.0, 0.0] # 命令速度0.5m/s                  │
    │   foot_frc = [55, 55]             # 双脚稳定接触                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 环境1: 较差表现                                                     │
    │   base_euler = [0.0, 0.52, 0.1]   # pitch=30°倾斜                   │
    │   base_lin_vel = [0.3, 0.1, 0.0]  # 速度不足,有侧向漂移              │
    │   commands = [0.5, 0.0, 0.0, 0.0] # 命令速度0.5m/s                  │
    │   foot_frc = [40, 60]             # 接触力不均衡                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    输出结果:# rewards 形状: (num_envs, 26)
    # ⭐ 关键步骤：gym_env_wrapper.step()按行求和（每个环境的所有奖励项相加）
    rew_buf = torch.clip(rew.sum(dim=1), min=0.)  
    # rew.sum(dim=1): 每行求和，形状从 (N,26) → (N,)
    # clip(min=0.): 确保奖励非负
    ┌─────────────────────────────────────────────────────────────────────┐
    │ rewards (简化为6项):                                                 │
    │   env0: [0.3, 0.95, 1.45, 2.2, 2.4, 0.65]  # 各项都优秀             │
    │   env1: [0.3, 0.50, 0.51, 1.0, 2.0,-0.10]  # 倾斜和速度问题严重      │
    │   (列顺序: constant, base_heit, balance, fwd_vel, yaw_rat, lateral) │
    ├─────────────────────────────────────────────────────────────────────┤
    │ eval_rew:                                                           │
    │   env0: 4.8  # 评估分数高                                           │
    │   env1: 1.2  # 评估分数低                                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 训练中使用:                                                         │
    │   rew_buf = rewards.sum(dim=1).clip(min=0)                          │
    │   rew_buf = [4.5, 1.8]  # 每个环境的总奖励                          │
    │   → 用于计算优势估计 → 驱动策略梯度更新                              │
    └─────────────────────────────────────────────────────────────────────┘

【完整示例2: 平衡奖励计算详解】
    balance_rew = 0.5 * (base_heit_rew * exp(-clip(5/lin_vel_x_norm, 2, 8) * norm(base_euler[:, :2])) + 1)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 场景: 倾斜30°行走                                                   │
    │   base_heit_rew = 0.5    # 基座高度较低                              │
    │   lin_vel_x_norm = 0.7   # 归一化速度                                │
    │   倾斜角度 = 30° = 0.52rad                                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 计算步骤:                                                           │
    │   1. 惩罚系数 = 5 / 0.7 = 7.14                                      │
    │   2. clip(7.14, 2, 8) = 7.14  # 在范围内保持不变                    │
    │   3. 指数惩罚 = exp(-7.14 × 0.52) = exp(-3.71) ≈ 0.024              │
    │   4. balance_rew = 0.5 × (0.5 × 0.024 + 1) ≈ 0.51                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 物理意义:                                                           │
    │   - 倾斜越大, exp(-k×θ) 越小, balance_rew 越低                       │
    │   - 速度越慢, k=5/v 越大, 惩罚越严厉                                 │
    │   - 高度越低, base_heit_rew 越小, 整体奖励被削弱                       │
    │   - 最终范围: [0.5, 1.0], 0.5为保底奖励                              │
    └─────────────────────────────────────────────────────────────────────┘

【在训练流程中的位置】
    GymEnvWrapper.step()
         ↓
    BIRLTask.action(net_out) → 生成关节指令
         ↓
    LeggedRobotEnv.step_torques() → 执行物理仿真
         ↓
    BIRLTask.step() → 更新状态
         ↓
    BIRLTask.reward() ← 当前函数
         ↓
    Storage.add('rewards', rewards) → 存储经验
         ↓
    PPO.update() → 基于奖励计算优势并更新策略

【训练原理】
    奖励 → 优势估计(GAE) → 策略梯度 → 参数更新
    正奖励 → 正优势 → 增大动作概率
    负奖励 → 负优势 → 减小动作概率
    机器人通过最大化累积奖励学会稳定走路!
=============================================================================
"""
