import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class Environment():
    def __init__(self):
        self.data = pd.read_csv('data/train/sh.000001_上证综合指数.csv')
        self.data['pct_change'] = pd.Series(self.data['close']).pct_change()
        self.data = self.data.fillna(0)

        self.barpos = 0

        self.buy_fee_rate = 0.0003
        self.sell_fee_rate = 0.0013
        self.order_size = 100000  # 订单大小

        self.init = 1000000
        self.fund = 1000000
        self.position = 0  # 持有股份数
        self.market_value = 0

        self.balance = 1000000
        self.total_profit = 0
        self.day_profit = 0

    def reset(self):
        self.barpos = 0

        self.init = 1000000
        self.fund = 1000000
        self.position = 0  # 持有股份数
        self.market_value = 0

        self.balance = 1000000
        self.total_profit = 0
        self.day_profit = 0

        observation = list(self.data.iloc[self.barpos])
        observation.append(self.balance)
        observation.append(self.position)
        observation.append(self.fund)
        return (observation)
 
    def step(self, action):
        # 
        current_price = self.data['close'][self.barpos]
        self.day_profit = self.position * current_price * self.data['pct_change'][self.barpos]
        if action == 0:  # 买入100000元的股票
            if self.fund > self.order_size:
                buy_order = math.floor(self.order_size / self.data['close'][self.barpos] / 100) * 100
                self.position += buy_order
                trade_amount = buy_order * current_price
                buy_fee = buy_order * current_price * self.buy_fee_rate
                self.fund = self.fund - trade_amount - buy_fee
                print("buy:success")
            else:
                print("buy:not enough fund")

        elif action == 1:  # 卖出100000元的股票
            if self.position * current_price > self.order_size:
                sell_order = math.ceil(self.order_size / self.data['close'][self.barpos] / 100) * 100
                self.position -= sell_order
                sell_fee = sell_order * current_price * self.sell_fee_rate
                trade_amount = sell_order * current_price
                self.fund = self.fund + trade_amount - sell_fee
                print("sell:success")
            else:
                print("sell:not enough stock")

        else:  # 啥也不做
            print("pass")

        # 重新计算持仓状况，不考虑除权除息
        self.market_value = self.position * current_price
        self.balance = self.market_value + self.fund
        self.total_profit = self.balance - self.init
        self.barpos += 1

        observation_ = list(self.data.iloc[self.barpos])
        observation_.append(self.balance)
        observation_.append(self.position)
        observation_.append(self.fund)

        # 转换 observation_ 的日期字符串为数值形式
        observation_ = [float(val.replace('-', '')) if isinstance(val, str) and val.replace('-', '').isdigit() else val for val in observation_]

        reward = self.day_profit
        if self.barpos == len(self.data) - 1:
            done = True
        else:
            done = False

        return (observation_, reward, done)

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state.to(self.device)
        x = F.relu(self.fc1(state.to(T.float32)))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    # gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
    # epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions=3,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, input_dims=input_dims, n_actions=self.n_actions,
                                   fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def save_model(self):
        T.save(self.Q_eval, 'DNN_Params')

    def load_model(self):
        self.Q_eval = T.load('DNN_Params')
    
    # 存储记忆
    def store_transition(self, state, action, reward, state_, done):
        # 将日期字符串转换为数值形式，跳过无法转换为整数的字符串
        state = [int(val.replace('-', '')) if isinstance(val, str) and val.replace('-', '').isdigit() else val for val in state]
        state_ = [int(val.replace('-', '')) if isinstance(val, str) and val.replace('-', '').isdigit() else val for val in state_]

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        print("store_transition index:", index)




    # observation就是状态state
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # 随机0-1，即1-epsilon的概率执行以下操作,最大价值操作
            state = T.tensor(observation).to(self.Q_eval.device)
            # 放到神经网络模型里面得到action的Q值vector
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            # epsilon概率执行随机动作
            action = np.random.choice(self.action_space)
            print("random action:", action)
        return action
    
    # 从记忆中抽取batch进行学习
    def learn(self):
        # memory counter小于一个batch大小的时候直接return
        if self.mem_cntr < self.batch_size:
            print("learn:watching")
            return

        # 初始化梯度0
        self.Q_eval.optimizer.zero_grad()

        # 得到memory大小，不超过mem_size
        max_mem = min(self.mem_cntr, self.mem_size)

        # 随机生成一个batch的memory index，可重复抽取
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # int序列array，0~batch_size
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # 从state memory中抽取一个batch
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)  # 存储是否结束的bool型变量

        # action_batch = T.tensor(self.action_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # 第batch_index行，取action_batch列,对state_batch中的每一组输入，输出action对应的Q值,batchsize行，1列的Q值
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0  # 如果是最终状态，则将q值置为0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

env = Environment()
# gamma的折扣率它必须介于0和1之间。越大，折扣越小。这意味着学习，agent 更关心长期奖励。
# 另一方面，gamma越小，折扣越大。这意味着我们的 agent 更关心短期奖励（最近的奶酪）。
# epsilon探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作。
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, eps_end=0.03, input_dims=[11], lr=0.003)
profits, eps_history = [], []
n_games = 100 #训练局数

for i in range(n_games):
    profit = 0
    done = False
    observation = env.reset()
    while not done:
        print("barpos:", env.barpos)
        action = agent.choose_action(observation)
        observation_, reward, done = env.step(action)
        profit = env.total_profit
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

        #保存一下每局的收益，最后画个图
        profits.append(profit)
        eps_history.append(agent.epsilon)
        avg_profits = np.mean(profits[-100:])

        print('episode', i, 'profits %.2f' % profit,
              'avg profits %.2f' % avg_profits,
              'epsilon %.2f' % agent.epsilon)

        # 保持 x 和 profits 的长度相同
        x = [i for i in range(1, len(profits) + 1)]

        plt.plot(x, profits)
