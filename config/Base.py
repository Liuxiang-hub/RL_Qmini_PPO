"""
Base.py - 双足机器人基础配置文件

功能:
  定义双足机器人的所有物理参数、控制参数、训练参数等
  相当于机器人的"出厂说明书",告诉系统机器人的一切规格

在机器人走路中的作用:
  1. 定义机器人有几条腿、多少关节
  2. 定义每个关节的活动范围
  3. 定义PD控制器的参数
  4. 定义奖励函数怎么计算
  5. 定义训练时的各种超参数

为什么需要配置文件?
  1. 将代码和参数分离,方便调参
  2. 便于复现实验
  3. 可以快速切换不同的机器人配置

代码结构:
  ┌───────────────────────────────────────────────────────────────────────┐
  │                         Base 配置类                                   │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │   task: 任务配置                                                     │
  │   ├─ cfg: 任务名称(Base/BIRL)                                       │
  │   └─ 用于选择不同的训练任务                                         │
  │                              ↓                                       │
  │   runner: 训练循环配置                                               │
  │   ├─ max_iterations: 训练多少轮(5000)                               │
  │   ├─ num_envs: 并行环境数(4096)                                    │
  │   └─ episode_length_s: 每个episode多久(10秒)                       │
  │                              ↓                                       │
  │   policy: 策略网络配置                                              │
  │   ├─ hidden_layers: 网络结构(512, 256)                             │
  │   └─ activation: 激活函数(relu)                                     │
  │                              ↓                                       │
  │   algorithm: PPO算法配置                                            │
  │   ├─ learning_rate: 学习率(1e-3)                                    │
  │   ├─ eps_clip: PPO裁剪范围(0.2)                                   │
  │   └─ gae_lambda: GAE参数(0.95)                                     │
  │                              ↓                                       │
  │   action: 动作空间配置                                              │
  │   ├─ ref_joint_pos: 参考关节角度(站立姿态)                         │
  │   └─ use_increment: 是否增量控制                                    │
  │                              ↓                                       │
  │   pd_gains: PD控制器配置                                           │
  │   ├─ stiffness: 比例增益Kp                                         │
  │   └─ damping: 微分增益Kd                                           │
  │                              ↓                                       │
  │   init_state: 机器人初始状态                                         │
  │   ├─ num_legs: 腿的数量(2=双足!)                                 │
  │   └─ pos: 初始位置[0, 0, 0.45]                                    │
  │                              ↓                                       │
  │   domain_rand: 领域随机化配置(Sim2Real关键!)                        │
  │   ├─ randomize_friction: 摩擦随机化                                │
  │   ├─ push_robots: 周期性推动                                      │
  │   └─ delay_observation: 传感器延迟模拟                             │
  │                              ↓                                       │
  │   terrain: 地形配置                                                │
  │   ├─ mesh_type: 地形类型                                           │
  │   └─ terrain_proportions: 地形难度分布                             │
  │                              ↓                                       │
  │   sim: 物理仿真参数                                                │
  │   ├─ dt: 仿真时间步(0.001秒)                                     │
  │   └─ gravity: 重力加速度                                          │
  │                              ↓                                       │
  │   asset: 机器人模型配置                                            │
  │   ├─ file: URDF模型路径                                           │
  │   └─ foot_name: 足部检测点                                        │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

与其他文件的关系:
  - train.py: 读取本配置文件创建环境和训练
  - env/legged_robot.py: 使用本配置创建仿真
  - rl/alg/ppo.py: 使用algorithm配置
"""
import numpy as np
from math import pi


class SetDict2Class():
    """工具类:将字典属性设置到对象"""
    def set_dict(self, dict):
        for key, value in dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Base:
    """双足机器人基础配置类"""
    def __init__(self):
        super(Base, self).__init__

    class task(SetDict2Class):
        """任务配置"""
        cfg = 'Base'

    class viewer:
        """仿真器视角配置"""
        pos = [0, 0, 0.6]     # 相机位置 [x, y, z] 米
        lookat = [0., 0, 0.6]  # 相机看向的位置
        fixed_robot_id = 0     # 跟随的机器人ID
        fixed_offset = [0., 2., 0.6]  # 相机偏移量

    class runner(SetDict2Class):
        """训练循环配置"""
        seed = 1                                # 随机种子
        max_iterations = 5000                  # 最大训练迭代次数
        num_steps_per_env = 24                 # 每次更新前收集的步数
        save_interval = 200                    # 模型保存间隔
        num_envs = 4096                        # 并行环境数量(加速训练)
        env_spacing = 3.                       # 环境间距
        send_timeouts = True                   # 发送超时信息
        episode_length_s = 10                 # Episode最大时长(秒)

    class policy(SetDict2Class):
        """策略网络配置"""
        name = 'simple_policy'                # 策略网络名称
        num_actions = None                    # 动作维度(运行时计算)
        num_critic_obs = None                # Critic观测维度
        num_observations = None               # 观测维度
        hidden_layers = (512, 256)            # 隐藏层结构
        activation = 'relu'                   # 激活函数

    class algorithm():
        """PPO算法超参数"""
        value_loss_coef = 1.                  # 价值损失系数
        use_clipped_value_loss = True        # 是否clip价值损失
        eps_clip = 0.2                       # PPO clip范围
        entropy_coef = 0.0005                # 熵正则化系数
        num_learning_epochs = 3               # 每次更新的epoch数
        num_mini_batches = 4                 # 小批次数量
        learning_rate = 1e-3                 # 学习率
        schedule = 'adaptive'                # 学习率调度
        discount_factor = 0.995              # 折扣因子
        gae_lambda = 0.95                    # GAE参数
        desired_kl = 0.01                    # 目标KL散度
        max_grad_norm = 1.                   # 梯度裁剪阈值

    class action(SetDict2Class):
        """动作空间配置"""
        action_limit_up = None
        action_limit_low = None

        # 动作缩放范围(不同关节有不同的控制范围)
        # 前2个是通用速度控制,后10个是关节角度控制
        high_ranges = [3.] * 2 + [1.] * 10   # 动作上限
        low_ranges = [0.5] * 2 + [-1.] * 10  # 动作下限

        # 参考关节位置(初始站立姿态)
        ref_joint_pos = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]

        use_increment = True                  # 是否使用增量控制
        inc_high_ranges = [10.] * 10         # 增量上限
        inc_low_ranges = [-10.] * 10         # 增量下限

    class pd_gains(SetDict2Class):
        """PD控制器增益配置
        
        PD控制器是最基础的机器人控制方法:
        torque = Kp * (target_pos - current_pos) + Kd * (target_vel - current_vel)
        
        不同关节需要不同的增益:
        - hip_yaw: 髋关节偏航(左右转)
        - hip_roll: 髋关节横滚(侧摆)
        - hip_pitch: 髋关节俯仰(前后摆)  
        - knee: 膝关节
        - ankle: 踝关节
        """
        decimation = 15                       # 控制频率分频
        stiffness = {                         # 比例增益Kp
            'hip_yaw': 55., 
            'hip_roll': 105., 
            'hip_pitch': 75., 
            'knee': 45., 
            'ankle': 30.
        }
        damping = {                          # 微分增益Kd
            'hip_yaw': 0.3, 
            'hip_roll': 2.5, 
            'hip_pitch': 0.3, 
            'knee': 0.5, 
            'ankle': 0.25
        }

    class init_state(SetDict2Class):
        """机器人初始状态配置"""
        random_rot = True                    # 随机初始朝向
        num_legs = 2                         # 腿的数量(双足!)
        pos = [0., 0., 0.45]               # 初始位置 [x, y, z] 米
        rot = [0., 0., 0., 1.]             # 初始朝向(单位四元数)
        lin_vel = [0.] * 3                  # 初始线速度
        ang_vel = [0.] * 3                  # 初始角速度
        reset_joint_pos = [0.4, -0.1, -1.5, 1., -1.3, -0.4, 0.1, 1.5, -1., 1.3]  # 初始关节角度

    class domain_rand(SetDict2Class):
        """领域随机化配置
        
        为什么需要领域随机化?
          为了弥合仿真和真实世界的差距(sim2real)
          通过在训练时引入各种随机变化,让策略更加鲁棒
        """
        randomize_friction = True           # 随机化摩擦系数
        friction_range = [0.2, 1.5]          # 摩擦系数范围
        randomize_mass = True                # 随机化质量
        added_mass_range = [0.5, 1.5]        # 质量缩放范围
        added_inertia_range = [0.5, 1.5]    # 惯性张量缩放范围
        randomize_damping = True             # 随机化阻尼
        added_friction_range = [0.8, 1.2]    # 阻尼范围
        added_damping_range = [0.8, 1.2]     # 阻尼范围
        randomize_torque = True             # 随机化最大扭矩
        torque_range = [0.8, 1.2]           # 扭矩范围
        randomize_gains = True               # 随机化PD增益
        gains_range = [0.8, 1.2]            # 增益范围
        
        # 扰动设置
        push_robots = True                  # 周期性推动机器人
        max_push_vel_xy = 0.5               # 最大推动速度
        max_push_rate_xyz = 0.5             # 推动频率
        push_interval_s = 3.                # 推动间隔
        
        # 观测延迟(模拟真实传感器延迟)
        delay_observation = True
        delay_joint_ranges = [10, 40]       # 关节位置/速度延迟范围
        delay_rate_ranges = [20, 50]        # 关节速度延迟范围
        delay_angle_ranges = [20, 50]       # 机体角度延迟范围

    class noise_values(SetDict2Class):
        """传感器噪声配置"""
        randomize_noise = True              # 是否添加噪声
        use_state_filter = False            # 是否使用状态滤波器
        lin_vel = 0.3                       # 线速度噪声标准差
        gravity = 0.15                      # 重力方向噪声
        ang_vel = 0.3                       # 角速度噪声
        foot_frc = 5.                       # 足底力噪声
        dof_pos = 0.1                       # 关节位置噪声
        dof_vel = 1.2                       # 关节速度噪声
        base_acc = 3.                       # 加速度计噪声

    class command(SetDict2Class):
        """运动指令配置
        
        机器人需要跟踪的目标运动状态
        """
        curriculum = False                  # 课程学习
        max_curriculum = 1.
        num_commands = 4                    # 命令维度
        resampling_time = 5.                # 命令重采样时间间隔(秒)
        heading_command = False             # 是否使用航向角命令

        # 可命令的速度范围
        lin_vel_x_range = [-0.3, 0.7]      # 前进速度范围 [m/s]
        lin_vel_y_range = [-0., 0.]        # 侧向速度范围(双足通常为0)
        ang_vel_yaw_range = [-1, 1]        # 偏航角速度范围 [rad/s]
        heading_range = [0., pi]           # 航向角范围

    class terrain(SetDict2Class):
        """地形配置"""
        mesh_type = 'trimesh'               # 地形类型:none/plane/trimesh
        horizontal_scale = 0.1              # 地形水平分辨率 [m]
        vertical_scale = 0.01               # 地形垂直分辨率 [m]
        border_size = 5                     # 地形边界大小 [m]
        static_friction = 1.               # 静摩擦系数
        dynamic_friction = 1.               # 动摩擦系数
        restitution = 0.05                  # 弹性恢复系数
        
        # 粗糙地形配置
        measured_points_x = [-0.05, 0, 0.05]  # 高度测量点
        measured_points_y = [-0.05, 0, 0.05]
        curriculum = False
        measure_heights = False
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 3          # 起始地形难度
        terrain_length = 15                 # 地形长度 [m]
        terrain_width = 15                  # 地形宽度 [m]
        num_rows = 20                       # 地形行数(难度级别)
        num_cols = 20                      # 地形列数(地形类型)
        # 地形类型比例: [平滑坡度, 粗糙坡度, 上楼梯, 下楼梯, 离散地形]
        terrain_proportions = [1., 0., 0., 0., 0.]
        slope_treshold = 0.0               # 坡度阈值

    class sim:
        """物理仿真参数"""
        dt = 0.001                          # 物理仿真时间步长 [秒]
        substeps = 1
        up_axis = 1                         # 0=y轴向上, 1=z轴向上
        gravity = [0., 0., -9.81]          # 重力加速度 [m/s²]

        class physx:
            """NVIDIA PhysX物理引擎配置"""
            solver_type = 1                 # 0: PGS, 1: TGS求解器
            num_threads = 10                 # 物理计算线程数
            num_position_iterations = 4      # 位置迭代次数
            num_velocity_iterations = 0     # 速度迭代次数
            contact_offset = 0.01            # 接触偏移 [m]
            rest_offset = 0.0               # 静止偏移 [m]
            bounce_threshold_velocity = 0.5  # 弹跳阈值速度
            max_depenetration_velocity = 1.0  # 最大穿透恢复速度
            max_gpu_contact_pairs = 2 ** 23 # GPU接触对数量
            default_buffer_size_multiplier = 5
            contact_collection = 2          # 接触收集模式
            use_gpu = True                  # 是否使用GPU

    class asset:
        """机器人模型资源配置"""
        enable_bar = False
        file = "assets/q1/urdf/q1.urdf"   # URDF模型文件路径

        imu_name = "imu_in_torso"          # IMU传感器名称
        foot_name = ['ankle_pitch']        # 足部名称(用于检测触地)
        base_name = "base_link"            # 基座名称
        penalize_contacts_on = ["knee", "hip"]      # 惩罚接触的部位
        terminate_after_contacts_on = ["knee", "base", "hip"]  # 接触后终止的部位
        disable_gravity = False
        collapse_fixed_joints = True       # 合并固定关节
        fix_base_link = False              # 固定基座(如在机架上)
        default_dof_drive_mode = 3         # 关节驱动模式(3=力矩模式)
        self_collisions = 0                # 自碰撞检测
        replace_cylinder_with_capsule = True  # 用胶囊体替代圆柱体
        flip_visual_attachments = False
        use_mesh_materials = True          # 使用材质颜色

        density = 0.1                      # 密度
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.
        thickness = 0.01