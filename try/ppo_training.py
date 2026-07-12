import os
from time import time
import torch
import gymnasium as gym                  # 导入gym库,用于创建环境
import numpy as np          # 导入numpy库,用于数值计算
from ppo_agent import PPOAgent  # 导入PPO智能体类



                                      # 0.创建环境
scenario = "Pendulum-v1"    # 定义环境名称
env = gym.make(scenario)     # 创建环境实例

NUM_EPISODES = 3000          # 定义训练的轮数
NUM_STEPS = 200           # 定义每轮的最大步骤数
BATCH_SIZE = 25          # 定义批次大小 用于训练时的批量更新 每个批次包含25个样本
UPDATE_INTERVAL = 50    # 定义更新间隔 用于训练时的批量更新 每50步更新一次参数 每个批次包含25个样本 分2个批次更新
STATE_DIM = env.observation_space.shape[0]  # 状态维度 分别是sin 角度,cos 角度,角速度w    3个维度
ACTION_DIM = env.action_space.shape[0]  # 动作维度 一个维度  即力矩u的维度

# 定义所有轮的最优奖励变量
best_reward = -1000  # 初始化最佳奖励为-1000  用于记录训练过程中出现的最佳奖励  用于判断是否需要保存模型参数

# 保存模型的文件路径
current_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
current_time = time.strftime("%Y%m%d-%H%M%S")  # 获取当前时间,格式为年月日-时分秒
model_path = current_path + "/models/" + f"ppo_actor_{current_time}.pth"  # 定义模型参数保存的文件路径,包含当前时间

# 初始化奖励缓冲区为空数组 用于存储每轮的总奖励
REWARD_BUFFER = np.empty(NUM_EPISODES)  # 初始化奖励缓冲区为空数组 用于存储每轮的奖励




                                      # 1. 创建PPO智能体实例
agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)  # 创建PPO智能体实例  TODO
# 参数: 状态维度, 动作维度, 批次大小   
# 输入为状态, 输出为动作

                                     # 2. 训练PPO智能体与环境交互并更新参数
for episode_i in range(NUM_EPISODES):  # 遍历训练的轮数
    state = env.reset()  # 重置环境,获取初始状态
    done = False        # 初始化环境未结束
    episode_reward = 0  # 初始化当前轮的奖励为0

    for step_i in range(NUM_STEPS):  # 在每轮中,智能体与环境交互NUM_STEPS步
        action,value = agent.get_action(state) #TODO  
         # get_action()方法根据当前状态state返回动作action和状态值value  
         # value是Critic网络的输出,用于评估动作的价值估计,即当前状态未来奖励的预估总和
        next_state, reward, done, _ = env.step(action)  # 执行动作,获取下一状态,奖励,是否结束等信息
        # step()方法执行动作action,返回下一状态next_state,奖励reward,是否结束done,其他信息_等信息
        episode_reward += reward  # 累积奖励
        done = True if step_i == NUM_STEPS - 1 else False  # 如果达到最大步骤数,则强制结束当前轮
        agent.replay_buffer.add_memory(state, action, reward, value, done)  
        # 将当前状态state、动作action、奖励reward、状态值value和是否结束done添加到智能体的回放缓冲区中
        state = next_state # 更新状态为下一状态
 
        # 3. 更新智能体参数
        if (step_i + 1) % UPDATE_INTERVAL == 0 or done == True:  # 每UPDATE_INTERVAL步更新一次智能体参数  或者环境结束时更新一次
            # update()方法根据回放缓冲区中的样本更新智能体的参数
            # 每个批次包含25个样本 分2个批次更新
            agent.update() # TODO  实现智能体参数的更新
            # 更新的是智能体的策略网络Actor和价值网络Critic的参数
            # Actor网络用于根据状态生成动作,他输出动作的概率分布 沿着让更优的动作概率增加的方向更新参数,使得智能体更倾向于选择那些带来更高奖励的动作
            # Critic网络用于评估动作的价值,他输出动作的价值估计 沿着让价值估计更准确的方向更新参数,使得智能体能够更好地评估动作的价值
            agent.replay_buffer.clear_memory()  # 清空回放缓冲区中的样本 

        # 4.择优保存模型参数
        if episode_reward > -100 and episode_reward > best_reward:  # 如果当前轮的奖励大于-100且大于最佳奖励   
            agent.save_policy()  # 保存当前智能体的模型参数 TODO  
            best_reward = episode_reward # 更新最佳奖励为当前轮的奖励
            torch.save(agent.policy.state_dict(), model_path)  # 官方方法保存模型参数 即Actor网络和Critic网络的权重和偏置参数
            print("Best reward: {:.2f}".format(best_reward))  # 打印当前最佳奖励    
            print("save model to {}".format(model_path))  # 打印保存模型的路径


        REWARD_BUFFER[episode_i] = episode_reward  # 将当前轮的奖励存储到奖励缓冲区中
        print("Episode {},  Reward {:.2f}".format(episode_i, episode_reward))  # 打印当前轮的奖励


        if done:  # 如果环境结束
            break  # 跳出当前轮的循环