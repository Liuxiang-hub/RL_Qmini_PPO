import torch
from torch import device, nn
from torch.distributions import Normal
import numpy as np
import torch.optim as optim

# 定义设备 如果有可用的GPU则使用GPU，否则使用CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

                                             # 1. 定义Actor网络
# 定义Actor网络class. Actor网络构建了四层全连接层 用于生成动作的概率分布 
# 类内方法forward 输入为状态 输出为动作的均值和标准差
# 类内方法select_action 输入为状态 调用forward方法根据动作的均值和标准差 从正态分布中采样动作  

class Actor(nn.Module):  # 类关系：Actor网络继承自nn.Module类
    def __init__(self, state_dim, action_dim, hidden_dim=256):     # 参数: 状态维度, 动作维度, 隐藏层维度默认256
        super(Actor, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 定义第一层全连接层 输入维度为状态维度 输出维度为隐藏层维度
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        # 定义第二层全连接层 输入维度为隐藏层维度 输出维度为隐藏层维度
        self.fc_mean = nn.Linear(hidden_dim, action_dim) # 定义第三层全连接层 输入维度为隐藏层维度 输出维度为动作维度
        self.fc_std = nn.Linear(hidden_dim, action_dim)  # 定义第四层全连接层 输入维度为隐藏层维度 输出维度为动作维度
        
        self.relu = nn.ReLU()  # 定义ReLU激活函数 Relu特点：输出非负，梯度恒为1
        self.tanh = nn.Tanh()  # 定义Tanh激活函数 Tanh特点：输出在-1到1之间，梯度在-1到1之间，非线性
        self.softplus= nn.Softplus()  # 定义Softplus激活函数 Softplus特点：输出非负，梯度在0到1之间，非线性


    def forward(self, x):  # 前向传播函数
        x = self.relu(self.fc1(x))  # 第一层全连接层后接ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层后接ReLU激活函数
        mean = self.tanh(self.fc_mean(x)) * 2  # 第三层全连接层后接Tanh激活函数  乘以2将输出范围调整到-2到2之间
        std = self.softplus(self.fc_std(x)) + 1e-3  # 第四层全连接层后接Softplus激活函数 输出动作的标准差 加上1e-3是为了避免标准差为0导致数值不稳定
        return mean, std      # 返回动作的均值和标准差 用于生成动作的概率分布
    
    
        # select_action()方法根据当前状态在动作空间 正态分布 中选择动作
        #直接选最大值 (greedy) ❌ → 容易陷入局部最优，缺乏探索
        #概率分布采样 ✅ → 有机会探索新动作，发现更好的策略
    def select_action(self, state):  # 根据当前状态选择动作

        # 情况	                        是否需要梯度
        # 训练 (更新权重)	           ✅ 需要，计算损失→反向传播
        # 推理/选择动作 (不更新权重)	❌ 不需要，节省显存、加速计算

        with torch.no_grad():  # 禁用梯度计算  避免计算梯度时的内存占用
            mu, sigma = self.forward(state)  # 前向传播获取动作的均值和标准差
            dist = torch.distributions.Normal(mu, sigma)  # 创建正态分布对象  
            action = dist.sample()  # 从正态分布中采样一个动作 意义：根据概率分布随机选择一个动作，提升探索能力
            action = action.clamp(-2, 2)  # 将采样的动作限制在-2到2之间  意义：确保动作在环境允许的范围内   

        return action  # 返回采样的动作和动作的对数概率 用于计算损失函数时使用



                                            # 2. 定义Critic网络  
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):     # 参数: 状态维度, 隐藏层维度默认256
        super(Critic, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 定义第一层全连接层 输入维度为状态维度 输出维度为隐藏层维度
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        # 定义第二层全连接层 输入维度为隐藏层维度 输出维度为隐藏层维度
        self.fc3 = nn.Linear(hidden_dim, 1)          # 定义第三层全连接层 输入维度为隐藏层维度 输出维度为1

        self.relu = nn.ReLU()  # 定义ReLU激活函数 Relu特点：输出非负，梯度恒为1

    def forward(self, x):  # 前向传播函数
        x = self.relu(self.fc1(x))  # 第一层全连接层后接ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层后接ReLU激活函数
        value = self.fc3(x)  # 第三层全连接层输出状态值value
        return value  # 返回状态值value 用于评估当前状态的价值
    # value是Critic网络的输出,用于评估动作的价值估计,即当前状态未来奖励的预估总和
    # value 会被用于计算优势函数,即当前动作的价值估计与真实奖励的差值
    # 优势函数用于衡量当前动作的好坏,如果优势函数为正,则说明当前动作比平均动作更好,反之则更差（平均·动作是指在当前状态下所有可能动作的平均表现）
    # 优势函数的计算公式为: A(t) = δ_t +  （γ入）δ_t+1 + (γ入)**2 δ_t+2 + ... + (γ入)**(T-t-1) δ_T-1  
    # 其中: δ_t = r + γV(t+1) - V(t) - V(t)  是当前动作的TD误差, γ是折扣因子, δ(t+1)是下一个状态的TD误差, T是当前轮的总步骤数, t是当前步骤数
    # 其中: r是当前奖励, γ是折扣因子, V(t+1)是下一个状态的价值估计, V(t)是当前状态的价值估计
    



  
                                             # 神经网络结构图
#┌─────────────────────────────────────────────────┬─────────────────────────────────────────────────┐
#│           🎯 策略网络 (Actor)                    │           💎 价值网络 (Critic)                  │
#├─────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
#│                                                 │                                                 │
#│  Input:  state_dim                              │  Input:  state_dim                              │
#│          ↓                                          ↓                                             │          
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     fc1       │ state_dim → hidden_dim      │     fc1       │ state_dim → hidden_dim          │
#│  │ Linear + ReLU │                             │ Linear + ReLU │                                 │
#│  └───────────────┘                             └───────────────┘                                 │   
#│          ↓                                            ↓                                             │
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     fc2       │ hidden_dim → hidden_dim     │     fc2       │ hidden_dim → hidden_dim        │
#│  │ Linear + ReLU │                             │ Linear + ReLU │                                 │
#│  └───────────────┘                             └───────────────┘                                 │
#│          ↓                                            ↓                                             │
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     fc3       │ hidden_dim → action_dim     │     fc3       │ hidden_dim → 1                  │
#│  │   Linear*2    │                             │   Linear      │                                 │
#│  └───────────────┘                             └───────────────┘                                 │
#│       ↓     ↓                                         ↓                                             │
#│  ┌───────────────┐                             ┌───────────────┐                                 │
#│  │     μ, σ      │  动作均值和标准差             │    value      │  状态价值 V(s)                  │
#│  │ Normal()采样   │  用于构建动作分布            │               │  评估当前状态好坏                │
#│  └───────────────┘                             └───────────────┘                                 │
#│          │                                          │                                             │
#└──────────┼──────────────────────────────────────────┼─────────────────────────────────────────────┘
#           │                                          │
#           ▼                                          ▼
#    ┌─────────────┐                            ┌─────────────┐
#    │   action    │                            │     V(s)    │
#    │  发送给dist  │                            │  计算优势函数│
#    │  供dist采样  │                            │  A(s) = r + γV(s') - V(s)
#    └─────────────┘                            └─────────────┘ 








                                             # 3. 定义回放缓冲区 
class ReplayMemory:  # 回放缓冲区   
    def __init__(self, batch_size):  # 参数: 批次大小
        self.state_capacity = []
        self.action_capacity = []
        self.reward_capacity = []
        self.value_capacity = []
        self.done_capacity = []
        self.batch_size = batch_size  # 批次大小 用于更新智能体参数时的样本采样


    def add_memory(self, state, action, reward, value, done):  # 添加记忆
        self.state_capacity.append(state)
        self.action_capacity.append(action)
        self.reward_capacity.append(reward)
        self.value_capacity.append(value)
        self.done_capacity.append(done)


    def sample(self):  # 样本采样
        # 打乱缓冲区中的样本顺序  避免训练时样本之间的相关性  提高训练效果
        num_states = len(self.state_capacity)  # 获取当前缓冲区中状态的数量 比如50
        batch_start_points = np.arange(0, num_states, self.batch_size)  # 计算每个批次的起始索引 比如[0, 25, 50]
        memory_indices  = np.arange(num_states)  # 获取当前缓冲区中所有样本的索引 比如[0, 1, 2, ..., 49]
        np.random.shuffle(memory_indices)  # 随机打乱样本的索引 顺序 打乱后可能是[49, 48, 47, ..., 0]
        batches = [memory_indices[i:i+self.batch_size] for i in batch_start_points]  
        # batches是一个列表，里面有num_states//batch_size 个元素，每个元素是一个长度为batch_size的样本索引列表
        # 把打乱后的样本索引根据批次batch_size大小进行分组
        # 方便后续根据这些索引从缓冲区中提取对应的状态、动作、奖励等数据进行训练

        return (
            np.array(self.state_capacity),  # 返回所有状态的数组 并转换为numpy数组 方便后续训练时使用
            np.array(self.action_capacity),  # 返回所有动作的数组 并转换为numpy数组 方便后续训练时使用
            np.array(self.reward_capacity), 
            np.array(self.value_capacity), 
            np.array(self.done_capacity),  
            batches  # 返回打乱后的批次索引列表 用于后续训练时按批次提取数据
        )
       

    def clear_memory(self):  # 清空记忆
        print("clear_memory")
        self.state_capacity = []
        self.action_capacity = []
        self.reward_capacity = []
        self.value_capacity = []
        self.done_capacity = []





                                            #  计算优势函数A例子
# 优势函数的计算公式为: A(t) = δ_t +  （γ入）δ_t+1 + (γ入)**2 δ_t+2 + ... + (γ入)**(T-t-1) δ_T-1  
                    # A(t) = A(t+1) * γ入 + δ_t
    # 其中: δ_t = r + γ*V(t+1) - V(t) - V(t) 
# 假设已经收集完一条T步的轨迹，拿到了所有数据
# rewards: [r0, r1, ..., rT-1]
# values:   [V(s0), V(s1), ..., V(sT)]   value来源：critic神经网络根据state输出的状态价值
# gamma (γ), lam (λ):  超参数，比如0.99和0.95

# 1. 先算所有TD误差δ_t，一步就能算完
#deltas = []
#for t in range(T):
#    delta = rewards[t] + gamma * values[t+1] - values[t]
#    deltas.append(delta)
#
## 2. 从后往前算优势A_t，也是一步循环
#advantages = [0] * T
## 最后一步的A，就是它自己的δ
#advantages[-1] = deltas[-1]
#for t in reversed(range(T-1)):  # 从T-2一直倒推到0
#    advantages[t] = deltas[t] + gamma * lam * advantages[t+1]

# 最后得到T个优势函数值advantages: [A0, A1, ..., A(T-1)]，每个A_t衡量了在状态s_t下动作a_t的好坏程度，正数表示比平均动作更好，负数表示更差
# A的作用是指导策略网络（Actor）更新参数，让它更倾向于选择那些带来更高奖励的动作，
# 同时也指导价值网络（Critic）更新参数，让它更准确地评估动作的价值。
# 优势函数A的值越大，表示该动作比平均动作更好，反之则更差。

# PPO 论文并行N个agent 即N个Actor网络，同时与环境交互,每个agent收集一条轨迹,每条轨迹包含T步数据,
# 总共收集N*T步数据,然后一起更新智能体参数,这样可以更高效地利用数据,加速训练过程。
# 具体是从NT个A(t)中随机抽取batch_size（M）个样本,然后根据这些样本更新智能体参数.




                                        # 4. 定义智能体类
class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        # 定义超参数
        self.LR_ACTOR = 1e-4  # Actor网络的学习率
        self.LR_CRITIC = 1e-4  # Critic网络的学习率
        self.GAMMA = 0.99      # 折扣因子 用于计算优势函数A和 TD误差δ_t 时的折扣因子
        self.LAMBDA = 0.95     # 折扣因子 用于计算优势函数A时的折扣因子
        self.K_EPOCHS = 10     # 重复训练的轮数
        self.EPOCHS = 10   
        # K epochs 就是：用同一批收集到的数据，重复训练 K 次。
        # 在你这篇 PPO 论文里，它是这么写的：
        # Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT
        # 翻译成大白话：
        # 针对策略参数 θ，用 K 个 epoch、小批量M 大小不超过总数据量 NT 的方式，优化PPO的代理目标函数 L。

        self.EPSILON = 0.2     # PPO的截断参数 用于限制策略更新时的步长
        # PPO的核心思想是限制策略更新的幅度，防止过大的更新导致性能崩溃。
        # 通过设置一个截断参数 ε，当新旧策略之间的概率比超过 1+ε 或小于 1-ε 时，就会对目标函数进行截断，
        # 限制更新的步长在合理范围内。

        self.actor = Actor(state_dim, action_dim).to(device)  # 创建Actor网络实例  # 创建Actor网络实例并将其移动到设备上
        self.old_actor = Actor(state_dim, action_dim).to(device)  # 创建旧的Actor网络实例  # 创建旧的Actor网络实例并将其移动到设备上
        self.critic = Critic(state_dim).to(device)             # 创建Critic网络实例
        self.state_dim = state_dim      # 状态维度
        self.action_dim = action_dim     # 动作维度 
        self.batch_size = batch_size     # 批次大小 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)  # Actor网络的优化器
        # Adam是一种自适应学习率优化器，适用于训练神经网络。它通过计算梯度的一阶矩估计和二阶矩估计来更新参数。
        # parameters()来自Actor网络实例，继承nn.Module类，返回Actor网络中所有可训练的参数
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)  # Critic网络的优化器
        #parameters()方法返回网络中所有可训练参数的迭代器，lr是学习率，Adam是一种自适应学习率优化算法，适用于训练神经网络
        self.replay_buffer = ReplayMemory(batch_size)  # 用于存储经验回放的缓冲区     


    def get_action(self, state):  # 根据当前状态获取动作
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # 将状态转换为浮点数张量并添加一个维度，
        #将其移动到设备上
        action = self.actor.select_action(state)  # 调用Actor网络的select_action方法根据当前状态获取动作
        value = self.critic.forward(state)  # 调用Critic网络的forward方法根据当前状态获取状态价值
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]  
    # 返回动作和状态价值 用于与环境交互和存储经验回放时使用
    # detach()方法用于从计算图中分离张量，返回一个新的张量，该张量与原始张量共享数据，但不参与梯度计算。
    # 这样可以将张量转换为NumPy数组，方便后续的数值计算和存储。

    def update(self):  # 更新智能体参数
        pass

    def save_policy(self):  # 保存模型参数
        pass