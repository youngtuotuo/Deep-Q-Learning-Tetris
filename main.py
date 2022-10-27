from tetris.game import Tetris
from tetris.constants import rows, cols
import torch
import torch.nn as nn
from src.rl import DQN, ReplayMemory, Transition
from random import random, randrange
import matplotlib.pyplot as plt

from itertools import count
import math
import pygame


def main(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Tetris()
    env.reset()
    states = torch.tensor(
        env.grid, dtype=torch.float16, device=device
    )
    print(states.shape)

    policy_net = DQN(rows, cols, env.n_actions).to(device)
    policy_net.half()

    target_net = DQN(rows, cols, env.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.half()
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters())
    criterion = nn.SmoothL1Loss()

    memory = ReplayMemory(opt.replay_memory_size)

    count_actions = 0

    episode_durations = []

    for epoch in range(opt.epochs):
        for t in count():
            env.tile_fall()
            sample = random()
            eps_threshold = opt.epsilon_end + (
                opt.epsilon_start - opt.epsilon_end
            ) * math.exp(-1.0 * count_actions / opt.epsilon_decay)
            count_actions += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    predictions = policy_net(states)
                    action = predictions.max(1)[1].view(1, 1)
            else:
                action = torch.tensor(
                    [[randrange(env.n_actions)]], device=device, dtype=torch.int64
                )

            reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            env.display()

            if done:
                env.reset()

            next_states = torch.tensor(
                env.grid,
                dtype=torch.float16,
                device=device,
            )

            # Store the transition in memory
            memory.push(states, action, next_states, reward)

            states = next_states

            if len(memory) >= opt.batch_size:
                transitions = memory.sample(opt.batch_size)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                states_batch = torch.cat(batch.states)
                action_batch = torch.cat(batch.action)
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
                next_states_batch = torch.cat(batch.next_states)
                next_state_values = target_net(next_states_batch).max(1)[0].detach()

                # Compute the expected Q values
                expected_state_action_values = (
                    next_state_values * opt.gamma
                ) + reward_batch

                # Compute Huber loss
                loss = criterion(
                    state_action_values, expected_state_action_values.unsqueeze(1)
                ) + 1e-6

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if done:
                episode_durations.append(t + 1)
                plt.figure(2)
                plt.clf()
                durations_t = torch.tensor(episode_durations, dtype=torch.float)
                plt.title("Training...")
                plt.xlabel("Episode")
                plt.ylabel("Duration")
                plt.plot(durations_t.numpy())
                # Take 100 episode averages and plot them too
                if len(durations_t) >= 100:
                    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                    means = torch.cat((torch.zeros(99), means))
                    plt.plot(means.numpy())

                plt.pause(0.001)  # pause a bit so that plots are updated
                break

        # Update the target network, copying all weights and biases in DQN
        if epoch % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("Complete")
    env.quit()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import argparse

    parser = argparse.ArgumentParser(description="DeepQ with Tetris\n\n")

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--replay_memory_size", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=200)
    parser.add_argument("--target_update", type=float, default=10)
    parser.add_argument(
        "--batch_size", type=int, default=128, help="The number of images per batch"
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument(
        "--saved_path", default="", type=str, help="model checkpoints path"
    )

    args = parser.parse_args()

    main(args)
