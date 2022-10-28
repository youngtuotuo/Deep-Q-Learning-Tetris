from tetris.game import Tetris
import torch
import torch.nn as nn
import os
import shutil
from src.rl import DeepQNetwork, ReplayMemory, Transition
from collections import deque
from random import random, randint, sample, randrange
from tetris.constants import play_width, play_height
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from itertools import count
import math


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))


    states_batch = torch.cat(batch.states).view(BATCH_SIZE, 4)
    action_batch = torch.round(torch.cat(batch.action)).to(torch.int64)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    predictions = policy_net(states_batch) 
    state_action_values = predictions.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_states_batch = torch.cat(batch.next_states).view(BATCH_SIZE, 4)
    next_state_values = target_net(next_states_batch).max()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Tetris()
    env.reset()
    states = torch.tensor(env.states, dtype=torch.float16, device=device)
    
    policy_net = DeepQNetwork(env.n_actions).to(device)
    policy_net.half()

    target_net = DeepQNetwork(env.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.half()
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters())
    criterion = nn.SmoothL1Loss()

    memory = ReplayMemory(opt.replay_memory_size)

    count_actions = 0

    for epoch in range(opt.epochs):
        for t in count():
            env.tile_fall()
            sample = random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * count_actions / EPS_DECAY)
            count_actions += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # print(f"{states=}")
                    predictions = policy_net(states)
                    # print(f"{predictions=}")
                    action = predictions.max().view(1, 1)
                    print(f"{t=}=>{action=}")
            else:
                action = torch.tensor([[randrange(env.n_actions)]], device=device, dtype=torch.int64)
                print(f"{t=}=>{action=}")


            reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            env.display()

            if not done:
                next_states = torch.tensor(env.states, dtype=torch.float16, device=device)
            else:
                next_states = None
                env.reset()

            # Store the transition in memory
            print(action)
            memory.push(states, action, next_states, reward)
            
            states = next_states

            optimize_model(policy_net, target_net, optimizer, memory, device)
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
       # Update the target network, copying all weights and biases in DQN
        if epoch % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


    print('Complete')
    env.quit()
    plt.ioff()
    plt.show()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepQ with Tetris\n\n")

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--replay_memory_size", type=int, default=10000)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", default="", type=str, help="model checkpoints path")
    
    args = parser.parse_args()

    main(args)
