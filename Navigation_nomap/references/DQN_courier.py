import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 环境参数
GRID_SIZE = 5
START_POS = (0, 0)
PACKAGE_POS = (2, 2)
TARGET_POS = (4, 4)
ACTIONS = ['up', 'down', 'left', 'right']  # 对应0,1,2,3

# DQN参数
BATCH_SIZE = 32
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000


class CourierEnv:
    """邮差任务网格环境"""

    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.agent_pos = START_POS
        self.has_package = False
        self.done = False

    def reset(self):
        self.agent_pos = START_POS
        self.has_package = False
        self.done = False
        return self._get_state()

    def _get_state(self):
        """状态包含：智能体坐标+是否携带包裹+目标坐标"""
        state = np.array([
            self.agent_pos[0] / GRID_SIZE,
            self.agent_pos[1] / GRID_SIZE,
            int(self.has_package),
            TARGET_POS[0] / GRID_SIZE,
            TARGET_POS[1] / GRID_SIZE
        ])
        return torch.FloatTensor(state)

    def step(self, action):
        x, y = self.agent_pos
        reward = 0

        # 执行动作
        if action == 0 and x > 0:  # down
            x -= 1
        elif action == 1 and x < GRID_SIZE - 1:  # up
            x += 1
        elif action == 2 and y > 0:  # 左
            y -= 1
        elif action == 3 and y < GRID_SIZE - 1:  # 右
            y += 1
        else:  # 无效移动
            reward -= 1

        self.agent_pos = (x, y)

        # 检查包裹拾取
        if not self.has_package and self.agent_pos == PACKAGE_POS:
            self.has_package = True
            reward += 10

        # 检查送达
        if self.has_package and self.agent_pos == TARGET_POS:
            reward += 50
            self.done = True
        else:
            reward -= 0.5  # 时间惩罚

        return self._get_state(), reward, self.done


class DQN(nn.Module):
    """深度Q网络"""

    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        self.buffer = deque(maxlen=MEMORY_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states),
            torch.tensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


def train():
    env = CourierEnv()
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    memory = ReplayBuffer()
    epsilon = EPSILON_START

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # ε-greedy动作选择
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = model(state.unsqueeze(0))
                    action = q_values.argmax().item()

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 存储经验
            memory.push(state, action, reward, next_state, done)
            state = next_state.clone()

            # 训练阶段
            if len(memory) > BATCH_SIZE:
                # 采样批次
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                # 计算目标Q值
                with torch.no_grad():
                    target_q = target_model(next_states).max(1)[0]
                    target_q = rewards + (1 - dones.float()) * GAMMA * target_q

                # 计算当前Q值
                current_q = model(states)
                see = actions.unsqueeze(1)
                current_q=current_q.gather(1, actions.unsqueeze(1))

                # 计算损失
                loss = nn.MSELoss()(current_q, target_q.unsqueeze(1))

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 更新目标网络
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        # ε衰减
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(f"Episode {episode}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}")


if __name__ == "__main__":
    train()