import numpy as np
import torch
from collections import deque


class CourierEnv:
    def __init__(self, dataset, num_landmarks=5, max_steps=200):
        self.dataset = dataset
        self.max_steps = max_steps
        self.current_node = None  # 当前节点ID
        self.landmarks = self._generate_landmarks(num_landmarks)  # 生成50个地标坐标
        self.target_landmark_idx = None  # 当前需要到达的地标索引
        self.visited_landmarks = []  # 已送达的地标
        self.step_count = 0
        self.last_action = None  # 上一个动作
        self.last_reward = 0.0  # 上一个奖励

    def _generate_landmarks(self, num):
        # 从数据集中随机选择num个节点作为地标
        landmark_ids = np.random.choice(self.dataset.node_ids, num, replace=False)
        return [self.dataset.metadata[self.dataset.metadata['node_id'] == id][['lat', 'lon']].values[0]
                for id in landmark_ids]

    def reset(self):
        # 随机选择初始节点和目标任务
        self.current_node = np.random.choice(self.dataset.node_ids)
        self.target_landmark_idx = np.random.randint(len(self.landmarks))
        self.visited_landmarks = []
        self.step_count = 0
        self.last_action = 0  # 初始动作设为0（无意义）
        self.last_reward = 0.0

        # 获取初始观察
        observation = self._get_observation()
        return observation

    def step(self, action):
        # 移动逻辑（简化为随机选择邻居）
        neighbors = self.dataset.graph.get(self.current_node, [])
        if len(neighbors) > 0 and action == 0:  # 假设动作0为“前进”
            self.current_node = np.random.choice(neighbors)

        # 计算奖励
        reward = 0
        done = False
        current_gps = self._get_node_gps(self.current_node)
        target_gps = self.landmarks[self.target_landmark_idx]

        # 是否到达目标地标
        if self._gps_distance(current_gps, target_gps) < 1e-4:
            reward += 100
            self.visited_landmarks.append(self.target_landmark_idx)
            # 分配新任务（如果还有未完成的地标）
            remaining = set(range(len(self.landmarks))) - set(self.visited_landmarks)
            if remaining:
                self.target_landmark_idx = np.random.choice(list(remaining))
            else:
                done = True
        else:
            # 基于距离减少的奖励
            prev_distance = self._gps_distance(self._get_prev_gps(), target_gps)
            new_distance = self._gps_distance(current_gps, target_gps)
            reward += (prev_distance - new_distance) * 10

        # 更新状态
        self.step_count += 1
        done = done or (self.step_count >= self.max_steps)
        self.last_action = action
        self.last_reward = reward

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # 获取当前视觉观察、地标编码、历史动作和奖励
        return {
            'view': self.dataset.get_view(self.current_node, 'front'),  # 假设Dataset有get_view方法
            'landmark_encoding': self._encode_landmarks(),
            'last_action': self.last_action,
            'last_reward': self.last_reward
        }

    def _encode_landmarks(self):
        # 对每个地标生成相对当前GPS的编码（方向和距离）
        current_gps = self._get_node_gps(self.current_node)
        encoding = []
        for lm in self.landmarks:
            dx = lm[0] - current_gps[0]
            dy = lm[1] - current_gps[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            direction = np.arctan2(dy, dx)  # 弧度
            encoding.extend([distance, np.cos(direction), np.sin(direction)])
        return torch.tensor(encoding, dtype=torch.float32)  # 形状 [50*3=150]

    def _get_node_gps(self, node_id):
        return self.dataset.metadata[self.dataset.metadata['node_id'] == node_id][['lat', 'lon']].values[0]

    def _gps_distance(self, gps1, gps2):
        return np.linalg.norm(gps1 - gps2)