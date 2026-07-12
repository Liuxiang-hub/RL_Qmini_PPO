"""
legged_robot.py - 双足机器人仿真环境核心类

功能:
  这是整个项目最重要的文件之一,负责创建和管理机器人的物理仿真环境
  相当于机器人的"数字孪生",在计算机中模拟机器人的物理行为

在机器人走路中的作用:
  1. 创建仿真世界(地面、地形、光照)
  2. 加载机器人URDF模型
  3. 模拟物理引擎(碰撞检测、力的传递)
  4. 读取传感器数据(IMU、关节角度、接触力)
  5. 执行控制指令(计算关节扭矩)
  6. 更新机器人状态

代码结构:
  ┌───────────────────────────────────────────────────────────────┐
  │                    LeggedRobotEnv 类结构                       │
  ├───────────────────────────────────────────────────────────────┤
  │                                                               │
  │   __init__(): 初始化                                          │
  │   ├─ 加载配置参数                                             │
  │   ├─ 设置设备(GPU/CPU)                                        │
  │   ├─ 创建仿真(create_sim)                                     │
  │   ├─ 初始化缓冲区(_init_buffers)                               │
  │   └─ 创建可视化窗口(可选)                                      │
  │                              ↓                                │
  │   create_sim(): 创建仿真环境                                   │
  │   ├─ 创建物理引擎                                             │
  │   ├─ 创建地形(平面/高度场/三角网格)                            │
  │   ├─ 加载机器人URDF模型                                       │
  │   └─ 创建多个并行环境                                         │
  │                              ↓                                │
  │   step_torques(): 执行一步仿真                                 │
  │   ├─ 计算PD控制器扭矩                                         │
  │   ├─ 执行物理仿真                                             │
  │   ├─ 刷新传感器数据                                           │
  │   └─ 处理噪声和滤波                                           │
  │                              ↓                                │
  │   _compute_torques(): PD控制器计算                             │
  │   └─ 根据目标角度计算关节扭矩                                  │
  │                              ↓                                │
  │   reset(): 重置环境                                           │
  │   ├─ 重置关节状态                                             │
  │   └─ 重置基座状态                                             │
  │                              ↓                                │
  │   _push_root_state_robots(): 随机推动机器人                     │
  │   └─ 增强策略鲁棒性                                           │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘

与其他文件的关系:
  - train.py: 创建本类实例进行训练
  - play.py: 创建本类实例进行测试
  - config/Base.py: 提供配置参数
  - env/tasks/birl_task.py: 使用本类获取状态计算奖励
"""
import collections
import math
import queue

# from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float, get_axis_params, to_torch
# import numpy as np
import os
import sys

from math import cos, sin, pi
from typing import Dict

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from config.Base import Base
from env.utils.math import quat_apply_yaw
import random
from scipy.spatial.transform import Rotation as R
from env.utils.terrain import Terrain
from collections import deque
from env.utils.delay_torch_deque import DelayDeque
import torch

from collections import OrderedDict


class LeggedRobotEnv:
    """
    双足机器人仿真环境类
    
    核心功能:
      1. 使用Isaac Gym创建物理仿真
      2. 管理多个并行环境(支持大规模强化学习训练)
      3. 提供传感器数据接口(IMU、关节、接触力)
      4. 实现PD控制器将角度指令转换为扭矩
      5. 支持领域随机化增强策略鲁棒性
    
    关键属性:
      - num_envs: 并行环境数量
      - device: 运行设备(cpu/cuda)
      - base_pos/quat/lvel/avel: 基座状态
      - joint_pos/vel: 关节状态
      - foot_frc: 足部接触力
      - torques: 关节扭矩
    """
    def __init__(self, cfg: Base, sim_params, physics_engine, sim_device, render, fix_cam, residual_cfg=None,
                 tcn_name=None, debug=False, epochs=1):
        """ 初始化仿真环境

        Args:
            cfg (Base): 配置文件对象,包含机器人参数、训练参数等
            sim_params (gymapi.SimParams): 物理仿真参数(时间步、重力等)
            physics_engine (gymapi.SimType): 物理引擎类型(必须是PhysX)
            sim_device (str): 仿真设备('cuda:0'或'cpu')
            render (bool): 是否启用可视化
            fix_cam (bool): 是否固定相机视角
            residual_cfg (dict, optional): 残差配置,用于自适应控制
            tcn_name (str, optional): TCN模型名称
            debug (bool, optional): 是否启用调试模式
            epochs (int, optional): 训练轮数
        """
        # 获取Gym接口
        self.gym = gymapi.acquire_gym()
        self.viewer = None
        self.cfg = cfg
        self.residual_cfg = residual_cfg if residual_cfg is not None else None
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.render = render
        self.debug = debug
        self.fix_cam = fix_cam
        self.tcn_name = tcn_name
        self.epochs = epochs
        
        # 获取腿的数量(双足机器人为2)
        self.num_legs = self.cfg.init_state.num_legs
        
        # 解析设备类型
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.device = self.sim_device if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline else 'cpu'
        self.graphics_device_id = -1 if not self.render else self.sim_device_id  # -1表示不渲染
        
        # 关节扭矩偏移(补偿重力等静态力)
        self.joint_tor_offset = torch.tensor([0.6, 1, 0., 0.7, 0.] + [-0.6, -1, -0., -0.7, 0.], dtype=torch.float, device=self.device)
        self.joint_vel_sign = torch.tensor([0., 1, 0., 0., 0.] * 2, dtype=torch.float, device=self.device)
        
        # 高度采样相关
        self.height_samples = None
        self.up_axis_idx = 2  # 2表示Z轴向上
        
        # 从配置获取关键参数
        self.num_envs = self.cfg.runner.num_envs
        self.num_observations = self.cfg.policy.num_observations
        self.num_actions = self.cfg.policy.num_actions
        
        # 状态缓冲区
        self.torques = None
        self.react_tau = None
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 时间参数
        self.dt = cfg.pd_gains.decimation * sim_params.dt
        self.max_episode_length_s = cfg.runner.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.dt)
        self.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)

        # 高精度状态(heading frame)
        self.foot_pos_hd = torch.zeros(self.num_envs, self.num_legs * 3, device=self.device, dtype=torch.float32)
        self.hand_pos_hd = torch.zeros(self.num_envs, self.num_legs * 3, device=self.device, dtype=torch.float32)
        self.base_pos_hd = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        # 如果地形类型不是高度场或三角网格,禁用课程学习
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False

        self.render_count = 0
        # 优化PyTorch JIT编译
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # 创建仿真、环境、缓冲区和可视化窗口
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()

        # 如果启用渲染,创建可视化窗口
        if self.render:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # 设置相机位置跟随机器人
            fixed_robot_id = self.cfg.viewer.fixed_robot_id
            fixed_robot_pos = self.base_pos.cpu().numpy().copy()[fixed_robot_id]
            fixed_robot_pos0 = self.base_pos.cpu().numpy().copy()[fixed_robot_id]
            fixed_robot_pos[0] = fixed_robot_pos0[0] - 1.5
            fixed_robot_pos[1] = fixed_robot_pos0[1] + 1.5
            fixed_robot_pos[2] = 0.5
            fixed_robot_pos0[2] = 0.5
            self.set_camera(fixed_robot_pos, fixed_robot_pos0)
            # 订阅键盘事件
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def quat_rotate_inv(self, q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def reset(self, env_ids):
        """ 重置指定环境
            
            重置关节状态、基座状态、历史缓冲区等,让机器人回到初始状态
            
        Args:
            env_ids (list[int]): 需要重置的环境ID列表
        """
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.episode_length_buf[env_ids] = 0
        
        # 如果启用扭矩随机化,重新采样扭矩增益
        if self.cfg.domain_rand.randomize_torque:
            self.tau_gains = torch_rand_float(self.cfg.domain_rand.torque_range[0],
                                              self.cfg.domain_rand.torque_range[1], (self.num_envs, self.num_dofs),
                                              device=self.device)
        
        # 如果启用增益随机化,重新采样PD增益
        if self.cfg.domain_rand.randomize_gains:
            self.p_gains_rand = torch_rand_float(self.cfg.domain_rand.gains_range[0],
                                                 self.cfg.domain_rand.gains_range[1], (self.num_envs, self.num_dofs),
                                                 device=self.device)
            self.d_gains_rand = torch_rand_float(self.cfg.domain_rand.gains_range[0],
                                                 self.cfg.domain_rand.gains_range[1], (self.num_envs, self.num_dofs),
                                                 device=self.device)
        
        # 重置历史缓冲区
        self.base_lin_vel_his.reset(env_ids, self.base_lin_vel[env_ids, :])
        self.base_ang_vel_his.reset(env_ids, self.base_ang_vel[env_ids, :])
        self.base_eul_his.reset(env_ids, self.base_euler[env_ids, :])

        self.joint_pos_his.reset(env_ids, self.joint_pos[env_ids, :].clone())
        self.joint_vel_his.reset(env_ids, self.joint_vel[env_ids, :].clone())
        self.joint_tau_his.reset(env_ids, self.react_tau[env_ids, :].clone())
        self.base_acc_his.reset(env_ids, self.base_acc[env_ids, :].clone())
        self.base_pos_hd_his.reset(env_ids, self.base_pos_hd[env_ids, :].clone())

        self.foot_frc_his.reset(env_ids, self.foot_frc[env_ids, :].clone())
        self.foot_pos_hd_his.reset(env_ids, self.foot_pos_hd[env_ids, :].clone())
        self.foot_vel_hd_his.reset(env_ids, self.foot_vel[env_ids, :].clone())

    def reset_counters(self, env_ids):
        """ 重置计数器
            
        Args:
            env_ids (list[int]): 需要重置计数器的环境ID列表
        """
        self.episode_length_buf[env_ids] = 0
        self.render_count = 0
        self.common_step_counter = 0

    def step_torques(self, joint_actions):
        """ 执行一步仿真,计算并应用关节扭矩
            
            这是仿真环境的核心方法,负责:
            1. 根据目标关节角度计算PD扭矩
            2. 执行物理仿真
            3. 刷新所有传感器数据
            4. 处理噪声和滤波
            5. 更新历史缓冲区
            
        Args:
            joint_actions (torch.Tensor): 目标关节角度(N, num_dofs)
        """
        # 1. 计算PD控制器扭矩
        self.torques = self._compute_torques(joint_actions).to(self.device)
        
        # 2. 如果启用扭矩随机化,乘以随机增益
        if self.cfg.domain_rand.randomize_torque:
            self.torques *= self.tau_gains
            self.torques = torch.clip(self.torques, -self.torque_limits, self.torque_limits)
        
        # 3. 设置关节扭矩并执行仿真
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        
        # 4. 刷新所有状态张量
        self.gym.refresh_dof_state_tensor(self.sim)          # 关节状态
        self.gym.refresh_actor_root_state_tensor(self.sim)   # 根节点状态
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 接触力
        self.gym.refresh_rigid_body_state_tensor(self.sim)   # 刚体状态
        self.gym.refresh_dof_force_tensor(self.sim)          # 关节力

        # 5. 从IMU获取基座状态(使用躯干上的IMU传感器)
        self.base_pos = self.rigid_body_param[:, self.imu_in_torso_indice, :3]  # 位置
        self.base_quat = self.rigid_body_param[:, self.imu_in_torso_indice, 3:7]  # 四元数
        self.base_lvel = self.rigid_body_param[:, self.imu_in_torso_indice, 7:10]  # 线速度
        self.base_avel = self.rigid_body_param[:, self.imu_in_torso_indice, 10:13]  # 角速度

        # 6. 计算加速度(数值微分)
        self.base_acc = (self.base_lvel - self.last_base_lvel) / self.cfg.sim.dt
        self.joint_acc = (self.joint_vel - self.last_dof_vel) / self.cfg.sim.dt
        self.last_base_lvel = self.base_lvel.clone()
        self.last_dof_vel = self.joint_vel.clone()

        # 7. 将四元数转换为欧拉角
        self.base_euler = self._get_euler_from_quat(self.base_quat)

        # 8. 将加速度从世界坐标系转换到机器人局部坐标系
        self.base_acc = (quat_rotate_inverse(self.base_quat, self.base_acc)).clip(min=-30., max=30)
        self.base_acc[:, [2]] += 9.8  # 加上重力加速度

        # 9. 将速度转换到机器人局部坐标系
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_lvel)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_avel)
        
        # 10. 计算投影重力(用于姿态估计)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # 11. 计算足部接触力(范数)
        self.foot_frc = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1).clip(min=0, max=1000)

        # 12. 计算足部欧拉角
        self.foot_euler = torch.cat(
            [torch.squeeze(self._get_euler_from_quat(self.rigid_body_param[:, foot, 3:7])) for foot in
             self.feet_indices], dim=-1).view(self.num_envs, -1)

        # 13. 计算足部位置
        self.foot_pos = torch.cat([torch.squeeze(self.rigid_body_param[:, foot, :3]) for foot in self.feet_indices],
                                  dim=-1).view(self.num_envs, -1)
        self.foot_pos[:, [2, 5]] -= 0.1  # 调整足部高度

        # 14. 计算heading frame下的状态(用于前进方向跟踪)
        base_yaw = self.base_euler[:, 2:3].clone()
        quat_tmp = quat_from_euler_xyz(torch.zeros_like(base_yaw), torch.zeros_like(base_yaw), base_yaw).squeeze(1)
        self.base_pos_hd = quat_rotate_inverse(quat_tmp, self.base_pos)
        
        # 15. 计算heading frame下的足部位置和速度
        self.foot_pos_hd = torch.cat(
            [torch.squeeze(quat_rotate_inverse(quat_tmp, self.foot_pos[:, 3 * foot:3 * foot + 3]))
             for foot in range(self.num_legs)], dim=-1).view(self.num_envs, -1)
        self.foot_vel = torch.cat([torch.squeeze(quat_rotate_inverse(quat_tmp, self.rigid_body_param[:, foot, 7:10]))
                                   for foot in self.feet_indices], dim=-1).view(self.num_envs, -1)

        # 16. 如果启用噪声,应用滤波和噪声
        if self.cfg.noise_values.randomize_noise:
            # 设置滤波权重
            if self.cfg.noise_values.use_state_filter:
                w4jp,w4jv,w4jt,w4bav,w4ba,w4blv,w4bac,w4ff=0.9,0.3,0.2,0.5,0.9,0.7,0.1,0.2
            else:
                w4jp, w4jv,w4jt, w4bav, w4ba, w4blv, w4bac,w4ff =(0 for _ in range(8))
            
            # 应用指数滤波
            joint_pos_filtered = self.exp_filter(self.joint_pos_his.delay(1), self.joint_pos + self.joint_pos_noise, w4jp)
            joint_vel_filtered = self.exp_filter(self.joint_vel_his.delay(1), self.joint_vel + self.joint_vel_noise, w4jv)
            joint_tau_filtered = self.exp_filter(self.joint_tau_his.delay(1), self.react_tau, w4jt)
            base_ang_vel_filtered = self.exp_filter(self.base_ang_vel_his.delay(1), self.base_ang_vel + self.ang_vel_noise, w4bav)
            base_euler_filtered = self.exp_filter(self.base_eul_his.delay(1), self.base_euler + self.gravity_noise, w4ba)
            base_acc_filtered = self.exp_filter(self.base_acc_his.delay(1), self.base_acc + self.base_acc_noise, w4bac)
            base_lin_vel_filtered = self.exp_filter(self.base_lin_vel_his.delay(1), self.base_lin_vel + self.lin_vel_noise, w4blv)
            foot_frc_filtered = self.exp_filter(self.foot_frc_his.delay(1), self.foot_frc + self.foot_frc_noise, w4ff)

            # 更新历史缓冲区
            self.joint_pos_his.append(joint_pos_filtered.clone())
            self.joint_vel_his.append(joint_vel_filtered.clone())
            self.joint_tau_his.append(joint_tau_filtered.clone())
            self.base_ang_vel_his.append(base_ang_vel_filtered.clone())
            self.base_eul_his.append(base_euler_filtered.clone())
            self.base_acc_his.append(base_acc_filtered.clip(min=-30., max=30).clone())
            self.foot_frc_his.append(foot_frc_filtered.clip(min=-1000., max=1000).clone())
            self.base_lin_vel_his.append(base_lin_vel_filtered.clone())
        else:
            # 不启用噪声时直接记录
            self.joint_pos_his.append(self.joint_pos.clone())
            self.joint_vel_his.append(self.joint_vel.clone())
            self.joint_tau_his.append(self.react_tau.clone())
            self.base_ang_vel_his.append(self.base_ang_vel.clone())
            self.base_eul_his.append(self.base_euler.clone())
            self.base_acc_his.append(self.base_acc.clone())
            self.base_pos_hd_his.append(self.base_pos_hd.clone())
            self.foot_frc_his.append(self.foot_frc.clone())
            self.base_lin_vel_his.append(self.base_lin_vel.clone())
        
        # 更新足部状态历史
        self.foot_pos_hd_his.append(self.foot_pos_hd.clone())
        self.foot_vel_hd_his.append(self.foot_vel.clone())
        
        # 如果是CPU模式,需要显式获取结果
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

    def step_states(self, iter=None):

        if self.cfg.terrain.mesh_type == 'plane':
            self.cfg.terrain.measure_heights = False
            self.foot_scanned_height = self.foot_pos.clone()
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_root_state_robots(iter=iter)
        if self.render:
            if self.render_count % 2 == 0:
                if self.fix_cam:
                    fixed_robot_id = self.cfg.viewer.fixed_robot_id
                    fixed_robot_pos = self.base_pos.clone().cpu().numpy().copy()[fixed_robot_id]
                    fixed_robot_pos0 = self.base_pos.clone().cpu().numpy().copy()[fixed_robot_id]
                    fixed_robot_pos[0] = fixed_robot_pos0[0] + 1.
                    fixed_robot_pos[1] = fixed_robot_pos0[1] - 1.
                    fixed_robot_pos[2] = 0.5
                    fixed_robot_pos0[2] = 0.5
                    # self.set_camera(fixed_robot_pos + np.array(self.cfg.viewer.fixed_offset), fixed_robot_pos)
                    self.set_camera(fixed_robot_pos, fixed_robot_pos0)
                self.rendering()
                self.render_count = 0
        self.render_count += 1
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_time_buf = self.episode_length_buf < 2
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length  # no terminal reward for time-outs

    def rendering(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.motor_action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.motor_action == "toggle_viewer_sync" and evt.value > 0:
                    self.render = not self.render
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            if self.render:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def set_camera(self, position, lookat):
        """ Set camera position and direction."""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callback --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs, 1),device='cpu')
                self.restitution_coeffs = torch_rand_float(0., 0.1, (self.num_envs, 1), device='cpu')
                # self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                props[s].restitution = self.restitution_coeffs[env_id]
                # print("props[s].friction", s, props[s].friction)
                # print("props[s].restitution", s, props[s].restitution)

        return props

    def _process_rigid_body_props(self, props):
        if self.cfg.domain_rand.randomize_mass:
            mrng = self.cfg.domain_rand.added_mass_range
            irng = self.cfg.domain_rand.added_inertia_range
            dm, di = np.random.uniform(mrng[0], mrng[1]), np.random.uniform(irng[0], irng[1])
        else:
            dm, di =  1,1
        for i in range(len(props)):
            props[i].inertia.x *= di
            props[i].inertia.y *= di
            props[i].inertia.z *= di
            if i == 0:
                props[i].mass *= 1.
            else:
                props[i].mass *= dm
        if self.cfg.domain_rand.randomize_mass:
            dm = np.random.uniform(-props[0].mass*0.6, props[0].mass*0.7)
            props[0].mass += dm
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
        return props

    def _compute_torques(self, joint_action):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        error = joint_action - self.joint_pos
        torques = self.p_gains * self.p_gains_rand * error + self.d_gains - self.d_gains_rand * self.joint_vel + self.joint_tor_offset - 3.5 * torch.sign(self.joint_vel) * self.joint_vel_sign
        return torch.clip(torques, - self.torque_limits, self.torque_limits).view(self.torques.shape)

    # ------------- Reset --------------
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if not self.render and self.cfg.init_state.random_rot:
            reset_dof_pos_noise = 0.1 * (2 * torch.ones_like(self.joint_pos[env_ids]) - 1.)
            reset_dof__vel_noise = 2. * (2 * torch.ones_like(self.joint_vel[env_ids]) - 1.)
        else:
            reset_dof_pos_noise = 0
            reset_dof__vel_noise = 0
        self.joint_pos[env_ids] = self.reset_joint_pos[env_ids, :] + reset_dof_pos_noise
        self.joint_vel[env_ids] = self.reset_joint_vel[env_ids, :] + reset_dof__vel_noise
        self.last_dof_vel[:] = self.joint_vel[:].clone()

        env_id_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_id_int32),
                                              len(env_id_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        init_rpy = 0.2 * (2 * torch.rand_like(self.base_euler) - 1.) if self.cfg.init_state.random_rot else torch.zeros_like(self.base_euler)
        self.rot = quat_from_euler_xyz(init_rpy[:, 0], init_rpy[:, 1], init_rpy[:, 2])

        if self.cfg.init_state.random_rot:
            base_init_state_list = self.cfg.init_state.pos + list(quat_from_euler_xyz(init_rpy[:, 0], init_rpy[:, 1], init_rpy[:, 2])[
                                                                      0].cpu().numpy()) + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        else:
            base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        reset_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        if self.custom_origins:
            self.root_states[env_ids] = reset_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = reset_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.cfg.init_state.random_rot:
            self.root_states[env_ids, 3:7] = self.rot[env_ids, :]

        # base velocities
        if not self.render:
            self.root_states[env_ids, 9:10] = torch_rand_float(-0.2, 0.2, (len(env_ids), 1),
                                                               device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
            self.root_states[env_ids, 7:9] = torch_rand_float(-0.5, 0.5, (len(env_ids), 2),
                                                              device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
            self.root_states[env_ids, 10:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 3),
                                                                device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        self.last_base_lvel[:] = self.root_states[:, 7:10]

        env_id_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_id_int32), len(env_id_int32))

    def _reset_dof_inertia(self):
        for i in range(self.num_envs):
            # self.dof_propss[i]['friction'] = np.random.uniform(self.cfg.domain_rand.added_friction_range[0], self.cfg.domain_rand.added_friction_range[1],
            #                                                    (1, self.num_dofs))
            # self.dof_propss[i]['damping'] = np.random.uniform(self.cfg.domain_rand.added_damping_range[0], self.cfg.domain_rand.added_damping_range[1], (1, self.num_dofs))
            # self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], self.dof_propss[i])
            rigid_body_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.actor_handles[i])
            rigid_body_props = self._process_rigid_body_props(rigid_body_props)
            self.gym.set_actor_rigid_body_properties(self.envs[i], self.actor_handles[i], rigid_body_props, recomputeInertia=True)

    def _push_root_state_robots(self, iter=None):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        ratio = 1. + min(iter / 3000., 0.5) if iter is not None else 1.
        max_vel = ratio * self.cfg.domain_rand.max_push_vel_xy
        max_rate = ratio * self.cfg.domain_rand.max_push_rate_xyz
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y
        self.root_states[:, [9]] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 1),
                                                    device=self.device)  # lin vel x/y
        self.root_states[:, 10:13] = torch_rand_float(-max_rate, max_rate, (self.num_envs, 3),
                                                      device=self.device)  # lin vel x/y
        self.root_states[:, 3:7] = self.rot[:, :]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))


    def create_sim(self):
        """ Creates simulation, terrain and environments
        Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        asset_path = self.cfg.asset.file
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print("body_names: {}".format(self.body_names))
        feet_names = [s for name in self.cfg.asset.foot_name for s in self.body_names if name in s]
        termination_contact_names = [s for name in self.cfg.asset.terminate_after_contacts_on for s in self.body_names
                                     if name in s]

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*to_torch(self.cfg.init_state.pos, device=self.device, requires_grad=False))

        self._get_env_origins()
        env_lower, env_upper = gymapi.Vec3(0., 0., 0.), gymapi.Vec3(0., 0., 0.)
        self.envs, self.actor_handles = [], []
        self.dof_propss, self.rigid_body_propss = [], []

        for i in range(self.num_envs):
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            pos += to_torch(self.cfg.init_state.pos, device=self.device, requires_grad=False)
            start_pose.p = gymapi.Vec3(*pos)
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i,
                                                 self.cfg.asset.self_collisions)  # todo!!! i
            if self.render:
                self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            dof_props = self._process_dof_props(dof_props_asset, i)
            if self.cfg.domain_rand.randomize_damping:
                dof_props['friction'] *= np.random.uniform(self.cfg.domain_rand.added_friction_range[0],
                                                           self.cfg.domain_rand.added_friction_range[1], self.num_dofs)
                dof_props['damping'] *= np.random.uniform(self.cfg.domain_rand.added_damping_range[0],
                                                          self.cfg.domain_rand.added_damping_range[1], self.num_dofs)
                # print("dof_props['friction']", dof_props['friction'])
                # print("dof_props['damping']", dof_props['damping'])

            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if self.cfg.domain_rand.randomize_mass:
                rigid_body_props = self._process_rigid_body_props(rigid_body_props)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, rigid_body_props, recomputeInertia=False)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.dof_propss.append(dof_props)
            self.rigid_body_propss.append(rigid_body_props)
        if self.render:
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, 2, 3))
            pass

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])
        self.imu_in_torso_indice = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.imu_name)

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols),
                                           rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.runner.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.react_tau = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dofs)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_param = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.rb_positions = gymtorch.wrap_tensor(rigid_body_state)[:, 0:3].view(self.num_envs, self.num_bodies, 3)

        self.joint_pos = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.joint_vel = self.dof_states.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.gravity_acc = to_torch([0., 0., 9.8], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.tau_gains = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                    requires_grad=False)

        self.p_gains_rand = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.d_gains_rand = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.base_pos = self.root_states[:, :3]  # positions
        self.base_quat = self.root_states[:, 3:7]  # quaternions
        self.base_lvel = self.root_states[:, 7:10]  # linear velocities
        self.base_avel = self.root_states[:, 10:13]  # angular velocities
        # torso
        # self.base_pos = self.rigid_body_param[:, self.torso_indice, :3] # positions
        # self.base_quat = self.rigid_body_param[:,self.torso_indice,  3:7]  # quaternions
        # self.base_lvel = self.rigid_body_param[:, self.torso_indice, 7:10]  # linear velocities
        # self.base_avel = self.rigid_body_param[:, self.torso_indice, 10:13]  # angular velocities

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_lvel)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_avel)
        self.base_euler = self._get_euler_from_quat(self.base_quat)

        self.base_acc = torch.zeros_like(self.base_lin_vel)  # linear velocities
        self.joint_acc = torch.zeros_like(self.joint_vel)  # angular velocities

        self.foot_pos = torch.cat([torch.squeeze(self.rigid_body_param[:, foot, :3]) for foot in self.feet_indices],
                                  dim=-1).view(self.num_envs, -1)
        self.foot_vel = torch.cat([torch.squeeze(self.rigid_body_param[:, foot, 7:10]) for foot in self.feet_indices],
                                  dim=-1).view(self.num_envs, -1)

        # self.foot_quat = torch.cat([torch.squeeze(self.rigid_body_param[:, foot, 3:7]) for foot in self.feet_indices], dim=-1).view(self.num_envs, -1)
        # self.l_foot_euler = quat_rotate_inverse(self.base_quat, self._get_euler_from_quat(self.foot_quat[:, :4]))
        # self.r_foot_euler = quat_rotate_inverse(self.base_quat, self._get_euler_from_quat(self.foot_quat[:, 4:]))
        self.foot_euler = torch.cat(
            [torch.squeeze(self._get_euler_from_quat(self.rigid_body_param[:, foot, 3:7])) for foot in
             self.feet_indices], dim=-1).view(self.num_envs, -1)

        self.foot_frc = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)

        self.base_lin_vel_his = DelayDeque(maxlen=1)
        self.base_pos_hd_his = DelayDeque(maxlen=1)
        self.foot_frc_his = DelayDeque(maxlen=1)

        self.base_eul_his = DelayDeque(maxlen=145)
        self.base_ang_vel_his = DelayDeque(maxlen=145)
        self.base_acc_his = DelayDeque(maxlen=145)

        self.joint_pos_his = DelayDeque(maxlen=135)
        self.joint_vel_his = DelayDeque(maxlen=135)

        self.joint_tau_his = DelayDeque(maxlen=1)

        self.foot_pos_hd_his = DelayDeque(maxlen=1)
        self.foot_vel_hd_his = DelayDeque(maxlen=1)

        for _ in range(self.joint_pos_his.maxlen):
            self.joint_pos_his.append(self.joint_pos.clone())
            self.joint_vel_his.append(self.joint_vel.clone())
            self.joint_tau_his.append(self.react_tau.clone())
        for _ in range(self.base_eul_his.maxlen):
            self.base_acc_his.append(self.base_acc.clip(min=-30., max=30.).clone())
            self.base_lin_vel_his.append(self.base_lin_vel.clone())
            self.base_eul_his.append(self.base_euler.clone())
            self.base_pos_hd_his.append(self.base_pos_hd.clone())

        for _ in range(self.base_ang_vel_his.maxlen):
            self.base_ang_vel_his.append(self.base_ang_vel.clone())
            self.foot_frc_his.append(self.foot_frc.clone())
        for _ in range(self.foot_pos_hd_his.maxlen):
            self.foot_pos_hd_his.append(self.foot_pos_hd.clone())
            self.foot_vel_hd_his.append(self.foot_vel.clone())

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.last_dof_vel = torch.zeros_like(self.joint_vel)
        self.last_base_lvel = torch.zeros_like(self.root_states[:, 7:10])

        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.foot_height_points = self._init_foot_height_points()
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            dof_name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.reset_joint_pos[i]
            found = False

            for gain_name in self.cfg.pd_gains.stiffness:
                if gain_name in dof_name:
                    found = True
                    if self.residual_cfg is not None:
                        self.p_gains[:, i] = self.residual_cfg.pd_gains.stiffness[gain_name]
                        self.d_gains[:, i] = self.residual_cfg.pd_gains.damping[gain_name]
                        # self.i_gains[:, i] = self.residual_cfg.pd_gains.integration[gain_name]
                    else:
                        self.p_gains[:, i] = self.cfg.pd_gains.stiffness[gain_name]
                        self.d_gains[:, i] = self.cfg.pd_gains.damping[gain_name]
            assert found, f'PD gain of {gain_name} joint was not defined'

        self.reset_joint_pos = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        self.reset_joint_vel = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device,
                                           requires_grad=False).repeat(self.num_envs, 1)
        self.reset_base_quat = self.base_quat
        init_rpy = 0.2 * (2 * torch.rand_like(self.base_euler) - 1.) if self.cfg.init_state.random_rot else torch.zeros_like(self.base_euler)
        self.rot = quat_from_euler_xyz(init_rpy[:, 0], init_rpy[:, 1], init_rpy[:, 2])

        if self.cfg.noise_values.randomize_noise:
            self.lin_vel_noise = self.cfg.noise_values.lin_vel * (2. * torch.rand_like(self.base_lin_vel) - 1.)
            self.gravity_noise = self.cfg.noise_values.gravity * (2. * torch.rand_like(self.base_euler) - 1.)
            self.ang_vel_noise = self.cfg.noise_values.ang_vel * (2. * torch.rand_like(self.base_ang_vel) - 1.)
            self.foot_frc_noise = self.cfg.noise_values.foot_frc * (2. * torch.rand_like(self.foot_frc) - 1.)
            self.joint_pos_noise = self.cfg.noise_values.dof_pos * (2. * torch.rand_like(self.joint_pos) - 1.)
            self.joint_vel_noise = self.cfg.noise_values.dof_vel * (2. * torch.rand_like(self.joint_vel) - 1.)
            self.base_acc_noise = self.cfg.noise_values.base_acc * (2. * torch.rand_like(self.base_acc) - 1.)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.border_size
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.as_tensor(torch.from_numpy(self.terrain.heightsamples)).view(self.terrain.tot_rows,
                                                                                                 self.terrain.tot_cols).to(
            self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.as_tensor(self.terrain.heightsamples, device=self.device)
        # self.height_samples = torch.as_tensor(torch.from_numpy(self.terrain.heightsamples)).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.as_tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x = torch.as_tensor(self.cfg.terrain.measured_points_x, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_foot_height_to_ground(self):
        x, y, z = self.foot_pos[:, [0, 3]], self.foot_pos[:, [1, 4]], self.foot_pos[:, [2, 5]]
        px = ((x.squeeze() + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
        py = ((y.squeeze() + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(px, 0, self.height_samples.shape[0] - 1)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 1)
        foot_sample = self.height_samples[px, py] * self.terrain.cfg.vertical_scale
        foot_heights = z - foot_sample
        return foot_heights

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) \
                     + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _init_foot_height_points(self):
        x = torch.as_tensor(self.cfg.terrain.measured_points_x, device=self.device)  #
        y = torch.as_tensor(self.cfg.terrain.measured_points_y, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_foot_height_points = 1  # grid_x.numel()
        foot_points = torch.zeros(self.num_envs, self.num_foot_height_points, 3, device=self.device,
                                  requires_grad=False)
        foot_points[:, :, 0] = grid_x.flatten()[0]
        foot_points[:, :, 1] = grid_y.flatten()[0]
        return foot_points


    def _get_euler_from_quat(self, quat):
        base_rpy = get_euler_xyz(quat)
        r = to_torch(self._from_2pi_to_pi(base_rpy[0]), device=self.device)
        p = to_torch(self._from_2pi_to_pi(base_rpy[1]), device=self.device)
        y = to_torch(self._from_2pi_to_pi(base_rpy[2]), device=self.device)
        return torch.t(torch.vstack((r, p, y)))

    def _from_2pi_to_pi(self, rpy):
        return rpy.cpu() - 2 * pi * np.floor((rpy.cpu() + pi) / (2 * pi))

    def exp_filter(self, history, present, weight):
        """
        exponential filter
        result = history * weight + present * (1. - weight)
        """
        return history * weight + present * (1. - weight)