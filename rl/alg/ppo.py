"""
ppo.py - PPO(近端策略优化)算法实现

功能:
  实现PPO强化学习算法,是机器人"学会走路"的核心学习机制
  相当于机器人的"大脑",负责根据经验调整策略

PPO算法是什么?
  PPO是一种强化学习算法,用于训练智能体学习最优策略
  核心思想:限制策略更新的幅度,避免一次性变化太大导致性能崩溃

为什么用PPO?
  1. 训练稳定:通过clipped surrogate objective限制更新
  2. 实现简单:相比TRPO,不需要复杂的约束优化
  3.效果好:在机器人控制任务中表现优异

在本项目中的作用:
  输入: 机器人当前状态观测 + 历史经验数据
  处理: 分析哪些动作获得了高奖励,调整网络参数
  输出: 更新后的策略网络(Actor)
  
核心公式:
  L^CLIP(θ) = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
  
  其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) 是概率比值
  A_t 是优势函数,衡量当前动作比平均水平好多少

代码结构:
  ┌───────────────────────────────────────────────────────────────────────┐
  │                         PPO 类结构                                     │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │   __init__: 初始化                                                   │
  │   ├─ 初始化Actor/Critic网络                                         │
  │   ├─ 创建Adam优化器                                                 │
  │   └─ 设置PPO超参数                                                  │
  │                              ↓                                        │
  │   init_storage(): 初始化经验回放缓冲区                                │
  │                              ↓                                        │
  │   act(): 策略采样                                                    │
  │   ├─ Actor网络前向传播                                               │
  │   ├─ 从分布中采样动作                                                │
  │   └─ 计算log概率(用于后续更新)                                      │
  │                              ↓                                        │
  │   process_env_step(): 处理一步经验                                   │
  │   ├─ 记录奖励和结束标志                                              │
  │   └─ 存入回放缓冲区                                                  │
  │                              ↓                                        │
  │   compute_returns(): 计算回报                                        │
  │   └─ GAE优势估计                                                    │
  │                              ↓                                        │
  │   update(): 策略更新(核心!)                                         │
  │   ├─ 小批次采样                                                     │
  │   ├─ 计算Clipped Surrogate Loss                                     │
  │   ├─ 计算价值损失                                                   │
  │   └─ 梯度更新                                                       │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

与其他文件的关系:
  - train.py: 调用PPO进行训练
  - model/simple_policy.py: PPO优化的Actor网络
  - rl/storage.py: 提供经验回放存储
"""
from rl.storage import Transition, RolloutStorage
import torch
import torch.nn as nn


class PPO:
    """
    PPO算法主类
    
    负责: 策略更新、价值估计、存储经验数据
    """
    def __init__(
            self,
            actor: torch.nn.Module,          # 策略网络(Actor)
            critic: torch.nn.Module,         # 价值网络(Critic)
            num_learning_epochs=1,           # 每次更新时对数据的重复利用次数
            num_mini_batches=1,             # 将数据分成多少个小批次
            learning_rate=1e-3,             # 学习率
            discount_factor=0.998,          # 折扣因子γ,考虑未来奖励的重要程度
            gae_lambda=0.95,               # GAE参数,平衡偏差和方差
            value_loss_coef=1.0,            # 价值损失系数
            entropy_coef=0.0,               # 熵正则化系数(鼓励探索)
            max_grad_norm=1.0,             # 梯度裁剪阈值
            desired_kl=0.01,               # 目标KL散度(用于自适应学习率)
            eps_clip=0.2,                   # PPO的clip范围[1-ε, 1+ε]
            use_clipped_value_loss=True,    # 是否对价值损失也做clip
            schedule="fixed",               # 学习率调度策略
            device='cpu',
    ):
        # 将网络移到指定设备
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.device = device


        # PPO核心组件初始化
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )
        self.transition = Transition()  # 临时存储当前步的经验
        self.storage = None  # 经验回放缓冲区,稍后初始化

        # PPO超参数
        self.eps_clip = eps_clip            # 策略更新的clip范围
        self.num_learning_epochs = num_learning_epochs      # 重复更新次数
        self.num_mini_batches = num_mini_batches            # 小批次数量
        self.value_loss_coef = value_loss_coef            # 价值损失权重
        self.entropy_coef = entropy_coef                  # 熵正则化权重
        self.gamma = discount_factor                       # 折扣因子
        self.gae_lambda = gae_lambda                      # GAE参数
        self.max_grad_norm = max_grad_norm                 # 梯度裁剪阈值
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, critic_obs_shape, actor_obs_shape, action_shape):
        """
        初始化经验回放存储器
        
        参数:
          num_envs: 并行环境数量
          num_transitions_per_env: 每个环境收集的步数
          critic_obs_shape: Critic网络的观测维度
          actor_obs_shape: Actor网络的观测维度
          action_shape: 动作空间维度
        """
        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            critic_obs_shape, 
            actor_obs_shape,
            action_shape, 
            self.device
        )

    def act(self, obs, cri_obs):
        """
        策略网络生成动作
        
        流程:
        1. Actor网络根据观测生成动作分布
        2. 从分布中采样动作
        3. 记录这个动作的log概率(用于后续PPO更新)
        4. Critic网络评估当前状态的价值
        """
        # 确保观测是float32类型
        obs = obs.to(torch.float32)
        
        # Actor网络前向传播
        res = self.actor(obs)
        actions, dist = res['act'].detach(), res['dist']  # detach:不计算梯度
        
        # 记录当前transition的所有信息(用于后续存储)
        self.transition.observations = obs
        self.transition.critic_obs = cri_obs
        self.transition.actions = actions
        
        # 计算log概率: log π(a|s)
        # 这在PPO中很重要,因为要用新策略和旧策略的概率比
        self.transition.actions_log_prob = dist.log_prob(actions).sum(dim=-1).detach()
        
        # 记录分布的均值和标准差
        self.transition.action_mean = dist.mean.detach()
        self.transition.action_sigma = dist.stddev.detach()
        
        # Critic评估状态价值
        self.transition.values = self.critic(cri_obs).detach()
        
        return actions

    def process_env_step(self, rewards, dones, infos):
        """
        处理一步交互的结果
        
        参数:
          rewards: 获得的奖励
          dones: 是否结束标志
          infos: 额外信息字典
        
        流程:
        1. 记录奖励和结束标志
        2. 处理超时情况(用于bootstrap)
        3. 将transition存入经验回放缓冲区
        """
        # 克隆奖励和结束标志
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # 处理超时情况
        # 如果episode超时而不是真正结束,需要bootstrap
        if 'timeouts' in infos:
            # 加上超时后下一个状态的价值估计作为最后奖励的补充
            self.transition.rewards += self.gamma * infos['timeouts'] * self.transition.values.squeeze()

        # 将transition存入回放缓冲区
        self.storage.add_transitions(self.transition)
        
        # 清空transition,准备记录下一步
        self.transition.clear()

    def compute_returns(self, cri_obs):
        """
        计算回报和GAE优势估计
        
        为什么需要GAE?
          优势函数A(s,a) = Q(s,a) - V(s) 衡量动作a比平均水平好多少
          GAE是一种平衡偏差和方差的优势估计方法
        
        公式:
          A_t^GAE = Σ_l=0^T (γλ)^l * δ_{t+l}
          其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        cri_obs = cri_obs.to(torch.float32)
        
        # 获取最后状态的价值估计(用于bootstrap)
        last_values = self.critic(cri_obs).detach()
        
        # 调用存储器的GAE计算方法
        self.storage.compute_returns(last_values, self.gamma, self.gae_lambda)

    def update(self):
        """
        PPO策略更新 - 核心训练步骤
        
        流程:
        1. 从存储中采样小批次数据
        2. 计算新的动作log概率和价值
        3. 计算PPO的clipped surrogate loss
        4. 计算价值损失
        5. 更新网络参数
        
        返回:
          mean_value_loss: 平均价值损失
          mean_surrogate_loss: 平均策略损失
          mean_kl: 平均KL散度(用于监控)
        """
        mean_surrogate_loss, mean_value_loss, mean_kl = 0., 0., 0.
        num_updates = self.num_learning_epochs * self.num_mini_batches
        
        # 创建小批次数据生成器
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, cri_obs_batch, actions_batch, target_values_batch, \
                advantages_batch, returns_batch, old_logp_batch, \
                old_mu_batch, old_sigma_batch in generator:
            
            # ===== 前向传播 =====
            res = self.actor(obs_batch)
            act, dist = res['act'], res['dist']
            
            # 计算新策略下的log概率
            logp_batch = dist.log_prob(actions_batch).sum(dim=-1)
            mu_batch = dist.mean
            sigma_batch = dist.stddev
            entropy_batch = dist.entropy().sum(dim=-1)
            
            # Critic评估
            value_batch = self.critic(cri_obs_batch)
            
            # ===== 自适应学习率调整 =====
            # 如果KL散度太大,减小学习率;太小则增大学习率
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    # 计算新旧策略的KL散度
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(2e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # ===== 计算PPO损失 =====
            # 概率比值: r(θ) = π_θ(a) / π_θ_old(a)
            ratio = (logp_batch - old_logp_batch.squeeze()).exp()
            
            # Clipped surrogate objective
            surr1 = ratio * advantages_batch.squeeze()           # 未clipped的损失
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_batch.squeeze()  # clipped的损失
            surrogate_loss = -torch.min(surr1, surr2).mean()    # 取较小值并取负(因为是梯度上升)
            
            # ===== 价值损失 =====
            if self.use_clipped_value_loss:
                # 价值函数也做clipped
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.eps_clip, self.eps_clip)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ===== 总损失 =====
            # 策略损失 + 价值损失 - 熵正则化(鼓励探索)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            
            # ===== 梯度更新 =====
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪,防止梯度爆炸
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            # 累加损失用于返回平均值
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.desired_kl != None and self.schedule == 'adaptive':
                mean_kl += kl_mean.item()

        # 清空存储器,准备下一轮数据收集
        self.storage.clear()
        
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates, mean_kl / num_updates