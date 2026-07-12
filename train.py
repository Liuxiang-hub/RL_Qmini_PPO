"""
train.py - PPO强化学习训练主程序

功能:
  这是整个项目的训练入口,负责将"让机器人走路"这个目标转化为具体的训练流程

在机器人走路中的作用:
  相当于"教练"角色,组织整个训练过程:
  1. 准备训练场地(创建仿真环境)
  2. 准备学习材料(初始化策略网络)
  3. 让机器人不断尝试(与环境交互)
  4. 告诉机器人哪些做得好、哪些做得不好(计算奖励)
  5. 指导机器人改进(PPO算法更新策略)
  6. 记录训练成果(保存模型)

完整流程:
  ┌─────────────────────────────────────────────────────────────┐
  │  1. 加载配置 (config/Base.py:机器人参数、PPO超参数)        │
  │                          ↓                                   │
  │  2. 创建仿真环境 (env/legged_robot.py:IsaacGym物理引擎)    │
  │                          ↓                                   │
  │  3. 初始化Actor网络 (model/simple_policy.py:策略大脑)       │
  │  4. 初始化Critic网络 (model:价值评估)                      │
  │                          ↓                                   │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │           训练主循环 (迭代5000次)                   │   │
  │  │                                                    │   │
  │  │  ┌─────────────────────────────────────────────┐  │   │
  │  │  │  经验收集阶段 (24步/迭代)                   │  │   │
  │  │  │  obs → Actor → action → env.step → reward  │  │   │
  │  │  └─────────────────────────────────────────────┘  │   │
  │  │                    ↓                              │   │
  │  │  ┌─────────────────────────────────────────────┐  │   │
  │  │  │  PPO算法更新 (3 epoch/迭代)                 │  │   │
  │  │  │  计算GAE → 策略梯度上升 → 保存模型          │  │   │
  │  │  └─────────────────────────────────────────────┘  │   │
  │  └─────────────────────────────────────────────────────┘   │
  │                          ↓                                │
  │  5. 输出训练好的模型: experiments/xxx/model/policy.pt    │
  └─────────────────────────────────────────────────────────────┘

使用方式:
  python train.py --config Base --name my_experiment --num_envs 4096
  
  --config: 配置文件名 (Base/BIRL)
  --name: 实验名称 (用于保存结果)
  --num_envs: 并行环境数量 (越多越快)

输出:
  - experiments/{name}/model/policy.pt: 最新模型
  - experiments/{name}/model/all/policy_*.pt: 定期保存的检查点
  - experiments/{name}/log/: TensorBoard训练日志

与其他文件的关系:
  - config/Base.py: 提供机器人配置参数
  - env/legged_robot.py: 提供仿真环境
  - model/simple_policy.py: 提供策略网络结构
  - rl/alg/ppo.py: 提供PPO算法实现
  - export_pt2onnx.py: 将训练好的模型导出用于机器人部署
"""
import importlib
import os
from os.path import join

from env.utils import get_args
from env.utils.helpers import update_cfg_from_args, class_to_dict, set_seed, parse_sim_params
from env import LeggedRobotEnv, GymEnvWrapper
from env.tasks import load_task_cls
from model import load_actor, load_critic
from rl.alg import PPO
import time
from collections import deque
import collections
import statistics
from utils.common import clear_dir
from utils.yaml import ParamsProcess
from isaacgym.torch_utils import *
from torch.utils.tensorboard import SummaryWriter
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train():
    """
    主训练函数
    
    训练流程:
    1. 加载配置和环境
    2. 初始化策略网络(Actor)和价值网络(Critic)
    3. 创建PPO算法实例
    4. 迭代训练:
       - 收集经验数据(与环境交互)
       - 使用PPO更新策略
       - 保存模型和记录日志
    """
    # 清理GPU缓存,释放未使用的显存
    torch.cuda.empty_cache()
    
    # ========== 1. 解析命令行参数和加载配置 ==========
    args = get_args()  # 获取命令行参数(配置文件名、训练设备等)
    device = args.rl_device  # 强化学习设备(CPU/GPU)
    
    # 动态导入配置文件(如config.Base或config.BIRL)
    cfg = getattr(importlib.import_module('.'.join(['config', args.config])), args.config)
    
    # 用命令行参数覆盖配置文件中的某些设置
    cfg = update_cfg_from_args(cfg, args)
    
    # 允许命令行指定并行环境数量(默认使用配置文件中的值)
    cfg.runner.num_envs = args.num_envs if args.num_envs is not None else cfg.runner.num_envs
    # ========== 2. 创建实验目录和日志记录器 ==========
    # 实验结果保存在 experiments/{实验名}/ 目录下
    exp_dir = join('experiments', args.name)
    model_dir = join(exp_dir, 'model')  # 模型保存目录
    os.makedirs(model_dir, exist_ok=True)  # 创建目录(已存在则跳过)
    
    # 保存所有检查点的子目录
    all_model_dir = join(exp_dir, 'model', 'all')
    os.makedirs(all_model_dir, exist_ok=True)
    
    # TensorBoard日志目录,用于可视化训练过程
    log_dir = join(exp_dir, 'log')
    clear_dir(log_dir)  # 清空旧日志
    writer = SummaryWriter(log_dir, flush_secs=10)
    # ========== 3. 配置训练参数 ==========
    num_steps_per_env = cfg.runner.num_steps_per_env  # 每次策略更新前收集的步数
    num_learning_iterations = cfg.runner.max_iterations  # 总训练迭代次数
    set_seed(seed=None)  # 设置随机种子(确保可复现性)
    
    # 解析IsaacGym物理仿真参数
    # ========== 4. 创建仿真环境 ==========
    # LeggedRobotEnv: 基于IsaacGym的腿式机器人仿真环境
    env = LeggedRobotEnv(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,  # 物理引擎选择
        sim_device=args.sim_device,            # 仿真设备
        render=args.render,                    # 是否渲染图形
        fix_cam=args.fix_cam                  # 固定相机视角
    )
    
    # 加载任务(训练目标:行走、平衡等)
    task = load_task_cls(cfg.task.cfg)(env)
    
    # GymEnvWrapper: 将仿真环境封装为Gym风格接口
    gym_env = GymEnvWrapper(env, task)
    
    # 计算观测空间维度和动作空间维度
    # pure_observation(): 获取单个观测向量
    # obs_history.maxlen: 观测历史帧数(用于时序策略如TCN)
    task.num_observations = len(gym_env.task.pure_observation()[0]) * gym_env.task.obs_history.maxlen
    task.num_actions = len(gym_env.task.action_low)  # 动作维度(关节数)

    # ========== 5. 处理配置文件 ==========
    # 将配置转换为字典格式,用于保存和日志记录
    cfg_dict = collections.OrderedDict()
    paramProcess = ParamsProcess()
    cfg_dict.update(paramProcess.class2dict(cfg))
    
    # 更新策略配置:观测维度、动作维度、 Critic观测维度
    cfg_dict['policy'].update({
        'num_observations': task.num_observations,
        'num_actions': task.num_actions,
        'num_critic_obs': len(gym_env.task.critic_observation()[0])
    })
    
    # 更新动作限制(关节角度上下限)
    # 从仿真环境中获取每个关节的实际限位
    cfg_dict['action'].update({
        'action_limit_low': env.dof_pos_limits[:, 0].cpu().numpy(),  # 各关节最小角度
        'action_limit_up': env.dof_pos_limits[:, 1].cpu().numpy()    # 各关节最大角度
    })
    
    # 动作缩放范围(用于增量控制)
    cfg_dict['action'].update({
        'action_scale_low': cfg.action.low_ranges[2:],  # 跳过前两个通用范围
        'action_scale_up': cfg.action.high_ranges[2:]
    })
    
    # 将完整配置保存为YAML文件,方便后续查看和复现
    paramProcess.write_param(join(model_dir, "cfg.yaml"), cfg_dict)

    # ========== 6. 创建策略网络和价值网络 ==========
    # Actor: 策略网络,输入状态输出动作(负责决策)
    # Critic: 价值网络,输入状态输出状态价值(负责评估)
    actor = load_actor(cfg_dict['policy'], device).train()
    critic = load_critic(cfg_dict['policy'], device).train()
    
    # ========== 7. 初始化PPO算法 ==========
    # 传入Actor、Critic和超参数,创建PPO优化器
    alg = PPO(actor, critic, device=device, **class_to_dict(cfg.algorithm))
    
    # 初始化经验回放缓冲区,用于存储训练数据
    # critic观测、actor观测、动作、奖励、价值估计(done=False时表示尚未计算出GAE)等信息
    alg.init_storage(
        cfg.runner.num_envs,                          # 并行环境数量
        num_steps_per_env,                           # 每个环境收集的步数
        [len(gym_env.task.critic_observation()[0])], # Critic观测维度
        [task.num_observations],                     # Actor观测维度
        [task.num_actions]                           # 动作维度
    )
    # ========== 8. 检查是否从中断点恢复训练 ==========
    if args.resume is not None:
        # 从指定实验目录加载已保存的模型
        resume_model_dir = join(join('experiments', args.resume), 'model')
        saved_model_state_dict = torch.load(join(resume_model_dir, 'policy.pt'))
        
        # 恢复Actor、Critic和优化器的状态
        alg.actor.load_state_dict(saved_model_state_dict['actor'])
        alg.critic.load_state_dict(saved_model_state_dict['critic'])
        alg.optimizer.load_state_dict(saved_model_state_dict['optimizer'])
        
        # 从中断处的迭代数继续训练
        current_learning_iteration = saved_model_state_dict['iteration']
    else:
        # 从头开始训练
        current_learning_iteration = 1

    # ========== 9. 初始化训练统计变量 ==========
    total_time, total_timesteps = 0., 0.  # 累计训练时间和步数
    
    # 总迭代次数 = 起始迭代 + 需要训练的新迭代数
    total_iteration = current_learning_iteration + num_learning_iterations
    
    # 使用deque作为滑动窗口,保存最近100个episode的统计数据
    rew_buffer = deque(maxlen=100)      # 累计奖励缓冲
    len_buffer = deque(maxlen=100)      # episode长度缓冲
    task_rew_buffer = deque(maxlen=100) # 任务奖励缓冲
    
    # 当前正在进行的episode的累计奖励和长度(每个环境一个值)
    cur_reward_sum = torch.zeros(cfg.runner.num_envs, dtype=torch.float, device=device)
    cur_task_rew_sum = torch.zeros(cfg.runner.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(cfg.runner.num_envs, dtype=torch.float, device=device)
    
    # ========== 10. 重置环境,获取初始观测 ==========
    obs, cri_obs = gym_env.reset(torch.arange(cfg.runner.num_envs, device=device))
    # ========== 11. 主训练循环 ==========
    for it in range(current_learning_iteration, total_iteration):
        start = time.time()  # 记录本次迭代开始时间
        
        # ------- 11.1 经验收集阶段 -------
        # 与环境交互num_steps_per_env步,收集训练数据
        for i in range(num_steps_per_env):
            # Actor根据当前观测生成动作
            act = alg.act(obs, cri_obs)
            
            # 环境执行动作,返回:新观测、奖励、是否结束、信息字典、任务奖励
            obs, cri_obs, rew, done, info, eval_rew = gym_env.step(act, it)
            
            # 将经验数据存入PPO的存储缓冲区
            alg.process_env_step(rew, done, info)
            
            # 累加奖励和步数
            cur_reward_sum += rew
            cur_task_rew_sum += eval_rew
            cur_episode_length += 1
            
            # 处理已结束的episode:记录统计数据并重置对应环境
            reset_env_ids = (done > 0).nonzero(as_tuple=False)[:, [0]].flatten()
            if len(reset_env_ids) > 0:
                # 将完成的episode数据加入缓冲区
                rew_buffer.extend(cur_reward_sum[reset_env_ids].cpu().numpy().tolist())
                task_rew_buffer.extend(cur_task_rew_sum[reset_env_ids].cpu().numpy().tolist())
                len_buffer.extend(cur_episode_length[reset_env_ids].cpu().numpy().tolist())
                # 重置已完成episode的累计值
                cur_reward_sum[reset_env_ids] = 0
                cur_task_rew_sum[reset_env_ids] = 0
                cur_episode_length[reset_env_ids] = 0
        
        # ------- 11.2 计算回报 -------
        # 使用GAE(广义优势估计)计算优势函数和回报
        alg.compute_returns(cri_obs)
        
        stop = time.time()
        collection_time = stop - start  # 经验收集耗时
        
        # ------- 11.3 策略更新阶段 -------
        start = stop
        # 调用PPO的update方法,执行策略梯度更新
        # 返回:价值损失、策略损失、KL散度
        mean_value_loss, mean_surrogate_loss, mean_kl = alg.update()
        
        # ------- 11.4 保存模型 -------
        # 构建模型状态字典(包含Actor、Critic、优化器状态和迭代数)
        saved_model_state_dict = {
            'actor': alg.actor.state_dict(),
            'critic': alg.critic.state_dict(),
            'optimizer': alg.optimizer.state_dict(),
            'iteration': current_learning_iteration,
        }
        
        # 保存最新模型到policy.pt(每次迭代都更新)
        try:
            torch.save(saved_model_state_dict, join(model_dir, 'policy.pt'))
        except OSError as e:
            print('Failed to save policy.')
            print(e)
        
        # 按间隔保存检查点(如policy_1000.pt、policy_2000.pt等)
        if it % cfg.runner.save_interval == 0:
            try:
                torch.save(saved_model_state_dict, join(all_model_dir, f'policy_{it}.pt'))
            except OSError as e:
                print('Failed to save policy.')
                print(e)
        
        stop = time.time()
        learn_time = stop - start  # 策略学习耗时
        # ========== 12. 统计和日志记录 ==========
        iteration_time = collection_time + learn_time  # 本次迭代总耗时
        total_time += iteration_time                    # 累计训练时间
        
        # 累计训练的步数 = 环境数 × 每环境步数 × 迭代数
        total_timesteps += num_steps_per_env * cfg.runner.num_envs
        
        # 计算FPS(每秒处理的样本数)
        fps = int(num_steps_per_env * cfg.runner.num_envs / iteration_time)
        
        # 获取Actor网络输出动作的标准差(反映探索程度)
        mean_std = alg.actor.std.mean()
        
        # 计算最近100个episode的平均统计值
        mean_reward = statistics.mean(rew_buffer) if len(rew_buffer) > 0 else 0.
        mean_task_reward = statistics.mean(task_rew_buffer) if len(task_rew_buffer) > 0 else 0.
        mean_episode_length = statistics.mean(len_buffer) if len(len_buffer) > 0 else 0.
        
        # ------- TensorBoard日志 -------
        # 训练奖励曲线
        writer.add_scalar('1:Train/mean_reward', mean_reward, it)
        writer.add_scalar('1:Train/mean_task_reward', mean_task_reward, it)
        writer.add_scalar('1:Train/mean_episode_length', mean_episode_length, it)
        writer.add_scalar('1:Train/mean_episode_time', mean_episode_length * gym_env.env.dt, it)
        
        # 损失函数曲线
        writer.add_scalar('2:Loss/value', mean_value_loss, it)
        writer.add_scalar('2:Loss/surrogate', mean_surrogate_loss, it)
        writer.add_scalar('2:Loss/learning_rate', alg.learning_rate, it)
        writer.add_scalar('2:Loss/mean_kl', mean_kl, it)
        writer.add_scalar('2:Loss/mean_noise_std', mean_std.item(), it)
        
        # 性能指标曲线
        writer.add_scalar('3:Perf/total_fps', fps, it)
        writer.add_scalar('3:Perf/collection_time', collection_time, it)
        writer.add_scalar('3:Perf/learning_time', learn_time, it)
        
        # ------- 控制台输出 -------
        # 打印格式: 实验名#迭代: 时间 col收集时间 lnt学习时间 nm帧率 m_kl...
        print(f"{args.name}#{it}:",
              f"{'t'} {total_time / 60:.1f}m({iteration_time:.1f}s)",
              f"col {collection_time:.2f}s",
              f"lnt {learn_time:.2f}s",
              f"nm {fps:.0f}",
              f"m_kl {mean_kl:.3f}",
              f"{'v_lss:'} {mean_value_loss:.3f}",
              f"{'a_lss:'} {mean_surrogate_loss:.3f}",
              f"l_n {int(mean_episode_length)}",
              f"total_rew {mean_reward:.2f}",
              f"task_rew {mean_task_reward:.2f}",
              sep='  ')
        
        # 更新当前迭代计数
        current_learning_iteration += 1


# ========== 程序入口 ==========
if __name__ == '__main__':
    # 当直接运行train.py时,调用train()函数开始训练
    # 避免在import时执行训练代码
    train()