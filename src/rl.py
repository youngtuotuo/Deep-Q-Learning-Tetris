import torch
import torch.nn as nn
from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("states", "action", "next_states", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DeepQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions),
        )

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x

class DQN2D(nn.Module):
    def __init__(self, in_channel, h, w, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        def conv2d_size_out(size, kernel_size = 4):
            return (size - kernel_size + 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 64

        self.head = nn.Sequential(
            nn.Linear(linear_input_size, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )
        self.float()

    def forward(self, x):
        x = self.net(x)
        return self.head(torch.flatten(x, 1))
