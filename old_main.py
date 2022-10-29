from tetris.game import Tetris
import torch
import torch.nn as nn
from torch.cuda import amp
import numpy as np
from src.rl import DeepQNetwork, ReplayMemory, Transition, DQN2D
from tetris.constants import play_width, play_height
from random import random, randrange
import matplotlib.pyplot as plt
import os

import datetime
from itertools import count
import math

plt.ion()


def main(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    exp_folder = os.path.join(
        os.getcwd(), f"exp-{datetime.datetime.now()}"[:-7].replace(" ", "-")
    )
    os.mkdir(exp_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = amp.GradScaler(enabled=(device.type != "cpu"))
    env = Tetris()
    env.reset()

    states = torch.tensor(env.info, dtype=torch.float, device=device)
    policy_net = DeepQNetwork(env.n_actions).to(device)
    target_net = DeepQNetwork(env.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters())
    criterion = nn.MSELoss()

    memory = ReplayMemory(opt.replay_memory_size)

    count_actions = 0

    epochs_durations = []
    losses = []
    rewards = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    for epoch in range(opt.epochs):
        for step in count():
            env.tile_fall()
            sample = random()
            eps_threshold = opt.epsilon_end + (
                opt.epsilon_start - opt.epsilon_end
            ) * math.exp(-1.0 * count_actions / opt.epsilon_decay)
            count_actions += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    predictions = policy_net(states)
                    action = predictions.max().view(1, 1)
            else:
                action = torch.tensor(
                    [[randrange(env.n_actions)]], device=device, dtype=torch.int64
                )

            reward, done = env.step(action.item())
            if done:
                reward -= 2
            reward = torch.tensor([reward], device=device)

            env.display()

            if not done:
                next_states = torch.tensor(env.info, dtype=torch.float, device=device)
            else:
                env.reset()
                next_states = torch.tensor(env.info, dtype=torch.float, device=device)

            # Store the transition in memory
            memory.push(states, action, next_states, reward)

            states = next_states

            if len(memory) >= opt.batch_size:
                transitions = memory.sample(opt.batch_size)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                states_batch = torch.cat(batch.states).view(opt.batch_size, 4)
                # states_batch = torch.cat(batch.states)
                action_batch = torch.round(torch.cat(batch.action)).to(torch.int64)
                reward_batch = torch.cat(batch.reward)

                with amp.autocast():
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                    # columns of actions taken. These are the actions which would've been taken
                    # for each batch state according to policy_net
                    predictions = policy_net(states_batch)
                    action_batch.clamp_(0, 3)
                    state_action_values = predictions.gather(1, action_batch)

                    # Compute V(s_{t+1}) for all next states.
                    # Expected values of actions for non_final_next_states are computed based
                    # on the "older" target_net; selecting their best reward with max(1)[0].
                    # This is merged based on the mask, such that we'll have either the expected
                    # state value or 0 in case the state was final.
                    next_states_batch = torch.cat(batch.next_states).view(
                        opt.batch_size, 4
                    )
                    next_state_values = target_net(next_states_batch).max()
                    # Compute the expected Q values
                    expected_state_action_values = (
                        next_state_values * opt.gamma
                    ) + reward_batch

                    # Compute Huber loss
                    loss = criterion(
                        state_action_values, expected_state_action_values.unsqueeze(1)
                    )

                # Optimize the model
                scaler.scale(loss).backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if done:
                epochs_durations.append(step + 1)
                losses.append(loss.item())
                rewards.append(reward.item())
                ax1.clear()
                ax2.clear()
                ax3.clear()
                ax1.set_ylabel("Steps")
                ax2.set_ylabel("Loss")
                ax3.set_ylabel("Reward")
                ax1.plot(np.array(epochs_durations))
                ax2.plot(np.array(losses))
                ax3.plot(np.array(rewards))

                plt.pause(0.001)  # pause a bit so that plots are updated
                fig.savefig(os.path.join(exp_folder, "statistics.png"))
                break
        # Update the target network, copying all weights and biases in DQN
        if epoch % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(
                policy_net.state_dict(), os.path.join(exp_folder, f"epoch-{epoch}-step-{step}.pt")
            )

    print("Complete")
    env.quit()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepQ with Tetris\n\n")

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--replay_memory_size", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_decay", type=float, default=200)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--target_update", type=float, default=10)
    parser.add_argument(
        "--batch_size", type=int, default=256, help="The number of images per batch"
    )
    parser.add_argument("--gamma", type=float, default=0.999)

    args = parser.parse_args()

    main(args)
