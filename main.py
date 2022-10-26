from tetris.game import Tetris
import torch
import torch.nn as nn
import os
import shutil
from src.rl import DeepQNetwork, ReplayMemory
from collections import deque
from random import random, randint, sample
from tetris.constants import play_width, play_height

from tensorboardX import SummaryWriter
from itertools import count
import math


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


def main(opt):

    if torch.cuda.is_availabel():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Tetris()
    env.reset()
    # state = torch.tensor(env.states).to(device)
    state = torch.tensor(env.window_array).to(device)
    
    policy_net = DeepQNetwork(env.n_actions).to(device)
    target_net = DeepQNetwork(env.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    replay_memory = ReplayMemory(opt.replay_memory_size)

    count_actions = 0

    for epoch in range(opt.epochs):
        for t in count():
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * count_actions / EPS_DECAY)
            count_actions += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[random.randrange(env.n_actions)]], device=device, dtype=torch.long)

            _, reward, done, _, _ = env.step(action.item())



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepQ with Tetris\n\n")

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--replay_memory_size", type=int, default=30000)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", default="", type=str, help="model checkpoints path")
    
    args = parser.parse_args()

    main(args)
