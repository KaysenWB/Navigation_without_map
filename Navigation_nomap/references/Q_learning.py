import numpy as np
import random

# 环境参数
GRID_SIZE = 5  # 网格大小 (5x5)
START_POS = (0, 0)  # 起点坐标
PACKAGE_POS = (2, 2)  # 包裹位置
TARGET_POS = (4, 4)  # 目标位置
ACTIONS = ['up', 'down', 'left', 'right']  # 动作空间

# Q-learning参数
EPISODES = 1000  # 训练轮数
ALPHA = 0.1  # 学习率
GAMMA = 0.99  # 折扣因子
EPSILON = 0.1  # 探索概率


class CourierEnv:
    """邮差任务网格环境"""

    def __init__(self):
        self.agent_pos = START_POS
        self.has_package = False
        self.done = False

    def reset(self):
        """重置环境状态"""
        self.agent_pos = START_POS
        self.has_package = False
        self.done = False
        return self._get_state()

    def _get_state(self):
        """返回当前状态（坐标 + 是否携带包裹）"""
        return (*self.agent_pos, int(self.has_package))

    def step(self, action):
        """执行动作，返回新状态、奖励、是否终止"""
        x, y = self.agent_pos
        reward = -0.1  # 时间惩罚（每步-0.1）

        # 执行动作
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < GRID_SIZE - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < GRID_SIZE - 1:
            y += 1
        else:
            reward = -1  # 无效动作惩罚

        self.agent_pos = (x, y)

        # 检查包裹拾取
        if not self.has_package and self.agent_pos == PACKAGE_POS:
            self.has_package = True
            reward = 10  # 拾取包裹奖励

        # 检查送达目标
        if self.has_package and self.agent_pos == TARGET_POS:
            self.done = True
            reward = 50  # 成功送达奖励

        return self._get_state(), reward, self.done


class QLearningAgent:
    """Q-learning智能体"""

    def __init__(self):
        # 初始化Q表：状态空间为 (x, y, has_package), 动作空间为4
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 2, len(ACTIONS)))

    def choose_action(self, state, epsilon):
        """ε-greedy选择动作"""
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        else:
            x, y, has_package = state
            return ACTIONS[np.argmax(self.q_table[x, y, has_package])]

    def update_q_table(self, state, action, reward, next_state):
        """更新Q表"""
        x, y, has_package = state
        action_idx = ACTIONS.index(action)

        # 下一状态的最大Q值
        next_x, next_y, next_has_package = next_state
        next_max_q = np.max(self.q_table[next_x, next_y, next_has_package])

        # Q-learning更新规则
        self.q_table[x, y, has_package, action_idx] = (1 - ALPHA) * self.q_table[x, y, has_package, action_idx] + \
                                                      ALPHA * (reward + GAMMA * next_max_q)


def train():
    env = CourierEnv()
    agent = QLearningAgent()

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            # 选择动作
            action = agent.choose_action(state, EPSILON)

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 更新Q表
            agent.update_q_table(state, action, reward, next_state)

            if done:
                break
            state = next_state

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # 训练后展示最优路径
    test_agent(agent)


def test_agent(agent):
    """测试训练后的智能体"""
    env = CourierEnv()
    state = env.reset()
    path = [state[:2]]
    total_reward = 0

    print("\n最优路径：")
    while True:
        action = agent.choose_action(state, epsilon=0)  # 关闭探索
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state[:2])

        print(f"位置: {state[:2]}, 动作: {action} -> 新位置: {next_state[:2]}, 奖励: {reward}")

        if done:
            print(f"成功送达！总奖励: {total_reward}")
            break
        state = next_state

    # 可视化路径
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=str)
    grid[:] = '·'  # 空地
    grid[PACKAGE_POS] = '📦'  # 包裹
    grid[TARGET_POS] = '🏁'  # 目标

    print("\n网格路径：")
    for x, y in path:
        if (x, y) == START_POS:
            grid[x][y] = '🚶'  # 起点
        else:
            grid[x][y] = '★'  # 路径点

    for row in grid:
        print(' '.join(row))


if __name__ == "__main__":
    train()