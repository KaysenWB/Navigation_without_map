
import torch.nn as nn
import torch
from collections import deque
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import *


class Self_Pos(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feats_in = args.feats_in
        self.feats_h = args.feats_h
        self.feats_out = args.feats_out
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.vision_encoder = args.vision_encoder
        self.pos_emb = Pos_Emb2(args)

        self.visual_encoder = nn.Sequential(
              nn.Conv2d(self.feats_in, self.feats_out//2, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.Conv2d(self.feats_h//2, self.feats_h, self.kernel_size//2, self.stride//2),
            nn.ReLU(),
            nn.Conv2d(self.feats_h, self.feats_h, self.kernel_size//2 -1, self.stride//4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.feats_h * 12 * 12, self.feats_h * 2)  # 假设CNN输出为64x12x12
        )

    def forward(self, current):

        if self.visual_encoder:
            enc = self.visual_encoder(current)
        else:
            enc = self.pos_emb(current)

        return enc


class LandmarkEncoder(nn.Module):
    """编码50个地标的相对位置信息"""

    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, landmark_encoding):
        # 输入: [batch_size, 50*3=150]
        # 每个地标的3维特征 -> 独立编码后求和
        batch_size = landmark_encoding.size(0)
        landmarks = landmark_encoding.view(batch_size, 50, 3)  # [B, 50, 3]
        encoded = self.encoder(landmarks)  # [B, 50, hidden_dim]
        return encoded.mean(dim=1)  # [B, hidden_dim]


class CourierAgent(nn.Module):
    def __init__(self, image_shape=(3, 256, 256), action_dim=3, lstm_hidden=256):
        super().__init__()
        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128)  # 假设CNN输出为64x12x12
        )

        # 地标编码器
        self.landmark_encoder = LandmarkEncoder(hidden_dim=64)

        # LSTM输入：视觉特征 + 地标编码 + 最后动作(one-hot) + 最后奖励
        self.lstm_input_dim = 128 + 64 + action_dim + 1
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden, batch_first=True)

        # 输出头
        self.value_head = nn.Linear(lstm_hidden, 1)  # 价值函数 V(s)
        self.policy_head = nn.Linear(lstm_hidden, action_dim)  # 策略 π(a|s)

    def forward(self, view, landmark_encoding, last_action, last_reward, hidden=None):
        batch_size = view.size(0)

        # 编码视觉
        visual_feat = self.visual_encoder(view)  # [B, 128]

        # 编码地标
        landmark_feat = self.landmark_encoder(landmark_encoding)  # [B, 64]

        # 处理历史动作和奖励
        last_action_onehot = F.one_hot(last_action, num_classes=3).float()  # [B, 3]
        last_reward = last_reward.unsqueeze(1)  # [B, 1]

        # 合并所有输入特征
        combined = torch.cat([visual_feat, landmark_feat, last_action_onehot, last_reward], dim=1)
        combined = combined.unsqueeze(1)  # LSTM需要序列维度 [B, 1, input_dim]

        # LSTM处理
        lstm_out, hidden_out = self.lstm(combined, hidden)  # lstm_out: [B, 1, hidden]
        lstm_out = lstm_out.squeeze(1)  # [B, hidden]

        # 输出价值与策略
        value = self.value_head(lstm_out)  # [B, 1]
        policy_logits = self.policy_head(lstm_out)  # [B, action_dim]

        return value, policy_logits, hidden_out