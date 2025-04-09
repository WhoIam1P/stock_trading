import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import warnings
from tqdm import tqdm
import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

# 使用CUDA如果可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class Agent:
    def __init__(self, state_size, action_size=3, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        
        # 超参数
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # 创建模型
        self.model = DQNModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def act(self, state):
        # 探索
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # 利用
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(device)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            
            with torch.no_grad():
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.model(next_state_tensor))
            
            # 前向传播获取当前Q值
            current_q = self.model(state_tensor)
            target_q = current_q.clone()
            target_q[0, action] = target
            
            # 计算损失并更新模型
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_q)
            loss.backward()
            self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def sigmoid(x):
    """缩放价格变化的sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def get_state(data, t, window_size):
    """生成状态表示"""
    if t < window_size:
        # 如果时间步小于窗口大小，填充前面的数据
        d = t - window_size + 1
        block = data[0:t+1]
        if d < 0:
            block = np.pad(block, (abs(d), 0), 'constant', constant_values=(block[0]))
    else:
        block = data[t-window_size+1:t+1]
    
    res = []
    for i in range(window_size - 1):
        res.append(sigmoid(block[i+1] - block[i]))
    
    return np.array([res])

def process_stock(ticker, save_dir, window_size=10, initial_money=10000, iterations=500):
    """处理单只股票的交易模拟"""
    try:
        # 创建必要的目录
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/trades", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/earnings", exist_ok=True)
        os.makedirs(f"{save_dir}/transactions", exist_ok=True)
        
        # 确保ticker不包含扩展名
        ticker = ticker.replace('.csv', '')
        
        # 读取数据
        file_path = f"{save_dir}/ticker/{ticker}.csv"
        print(f"正在读取文件: {file_path}")
        stock_data = pd.read_csv(file_path)
        
        if stock_data.empty:
            raise ValueError(f"股票 {ticker} 的数据为空")
        
        # 获取收盘价
        if 'Close' not in stock_data.columns:
            raise ValueError(f"股票数据中缺少Close列，可用列: {stock_data.columns.tolist()}")

        # 显示数据信息
        print(f"数据形状: {stock_data.shape}")
        print(f"包含列: {', '.join(stock_data.columns)}")
        
        close_prices = stock_data['Close'].values
        
        # 检查数据是否足够
        if len(close_prices) < window_size * 3:
            raise ValueError(f"数据样本太少，至少需要 {window_size * 3} 行数据")
            
        # 划分训练集和测试集
        split = int(0.7 * len(close_prices))
        train_data = close_prices[:split]
        test_data = close_prices[split:]
        
        print(f"训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")
        
        # 初始化智能体
        agent = Agent(window_size - 1)
        
        # 训练模式
        print(f"开始训练智能体，迭代 {iterations} 次...")
        for e in tqdm(range(iterations), desc="训练进度"):
            state = get_state(train_data, 0, window_size)
            total_profit = 0
            agent.inventory = []
            
            for t in range(1, len(train_data) - 1):
                action = agent.act(state)
                next_state = get_state(train_data, t, window_size)
                reward = 0

                # 买入
                if action == 1:
                    agent.inventory.append(train_data[t])
                
                # 卖出
                elif action == 2 and len(agent.inventory) > 0:
                    bought_price = agent.inventory.pop(0)
                    reward = max(0, train_data[t] - bought_price)
                    total_profit += train_data[t] - bought_price
                
                done = t == len(train_data) - 2
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                # 每隔100步或训练结束时才调用replay
                if t % 100 == 0 or done:
                    agent.replay()
                
                if done and e % 10 == 0:
                    tqdm.write(f"回合: {e+1}/{iterations}, 训练利润: {total_profit:.2f}")
        
        # 保存模型
        torch.save(agent.model.state_dict(), f"{save_dir}/models/{ticker}_dqn.pth")
        
        # 测试模式
        agent.is_eval = True
        agent.epsilon = 0  # 不再探索
        
        print("开始测试交易策略...")
        state = get_state(test_data, 0, window_size)
        total_profit = 0
        agent.inventory = []
        
        history = []
        balance = initial_money
        buy_dates = []
        sell_dates = []
        buy_prices = []
        sell_prices = []
        transactions = []
        
        for t in tqdm(range(1, len(test_data) - 1), desc="测试进度"):
            action = agent.act(state)
            next_state = get_state(test_data, t, window_size)
            
            # 计算当前持仓价值
            holding_value = sum(agent.inventory)
            current_value = balance + holding_value
            history.append(current_value)
            
            # 买入
            if action == 1 and balance > test_data[t]:
                agent.inventory.append(test_data[t])
                balance -= test_data[t]
                buy_dates.append(t)
                buy_prices.append(test_data[t])
                
                transactions.append({
                    'day': t,
                    'operate': '买入',
                    'price': test_data[t],
                    'investment': test_data[t],
                    'total_balance': balance + sum(agent.inventory)
                })
                
            # 卖出
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                balance += test_data[t]
                total_profit += test_data[t] - bought_price
                sell_dates.append(t)
                sell_prices.append(test_data[t])
                
                transactions.append({
                    'day': t,
                    'operate': '卖出',
                    'price': test_data[t],
                    'investment': test_data[t],
                    'total_balance': balance + sum(agent.inventory)
                })
            
            state = next_state
        
        # 计算最终价值和回报率
        final_value = balance + sum([test_data[-1]] * len(agent.inventory))
        invest_return = ((final_value - initial_money) / initial_money) * 100
        
        print(f"初始资金: ${initial_money:.2f}")
        print(f"最终资金: ${final_value:.2f}")
        print(f"总收益: ${final_value - initial_money:.2f}")
        print(f"投资回报率: {invest_return:.2f}%")
        print(f"买入次数: {len(buy_dates)}")
        print(f"卖出次数: {len(sell_dates)}")
        
        # 保存交易记录
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df.to_csv(f"{save_dir}/transactions/{ticker}_transactions.csv", index=False)
        else:
            # 创建一个空的交易记录文件
            pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance']
                        ).to_csv(f"{save_dir}/transactions/{ticker}_transactions.csv", index=False)
        
        # 生成交易图
        plt.figure(figsize=(15, 5))
        plt.plot(test_data, label='股票价格', color='black', alpha=0.5)
        
        if buy_dates:
            plt.scatter(buy_dates, [test_data[i] for i in buy_dates], marker='^', c='green', alpha=1, s=100, label='买入')
        if sell_dates:
            plt.scatter(sell_dates, [test_data[i] for i in sell_dates], marker='v', c='red', alpha=1, s=100, label='卖出')
        
        plt.title(f"{ticker} 交易记录")
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局，确保所有内容都能显示
        plt.savefig(f"{save_dir}/pic/trades/{ticker}_trades.png", dpi=300)
        plt.close()
        
        # 生成收益图
        plt.figure(figsize=(15, 5))
        plt.plot(history, label='投资组合价值', color='blue')
        plt.axhline(y=initial_money, color='r', linestyle='-', label='初始投资')
        plt.title(f"{ticker} 累计收益")
        plt.xlabel('时间')
        plt.ylabel('投资组合价值 ($)')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        plt.savefig(f"{save_dir}/pic/earnings/{ticker}_cumulative.png", dpi=300)
        plt.close()
        
        return {
            'total_gains': final_value - initial_money,
            'investment_return': invest_return,
            'trades_buy': len(buy_dates),
            'trades_sell': len(sell_dates)
        }
    
    except Exception as e:
        print(f"处理股票 {ticker} 出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 确保创建空文件以避免后续处理错误
        os.makedirs(f"{save_dir}/transactions", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/trades", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/earnings", exist_ok=True)
        
        # 创建空白图像 - 确保错误信息也能正确显示
        plt.figure(figsize=(15, 5))
        plt.title(f"{ticker} - 处理出错")
        plt.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pic/trades/{ticker}_trades.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(15, 5))
        plt.title(f"{ticker} - 处理出错")
        plt.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pic/earnings/{ticker}_cumulative.png", dpi=300)
        plt.close()
        
        # 创建空的交易记录
        pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance']
                   ).to_csv(f"{save_dir}/transactions/{ticker}_transactions.csv", index=False)
        
        # 返回空结果
        return {
            'total_gains': 0,
            'investment_return': 0,
            'trades_buy': 0,
            'trades_sell': 0
        }

def main():
    """主函数：执行所有股票的交易策略"""
    # 股票列表
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
        'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
        'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
    ]
    save_dir = 'results'
    # 处理每只股票
    for ticker in tickers:
        process_stock(ticker, save_dir)

if __name__ == "__main__":
    main()