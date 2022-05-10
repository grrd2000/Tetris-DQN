import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from src.model_new import DQNet, QTrainer
from src.tetris import Tetris
from src.tetris_new import Tetris_new
from src.tools import plot
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
RENDER = False
PLOT = True


class Agent:
    def __init__(self):
        self.n_epochs = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQNet(6, 256, 1)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_states(self, game):
        next_steps = game.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        return next_states

    def get_state(self, game):
        state = game.get_state_properties(game.board)
        state = np.array(state)

        return state

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # one tuple

    def train_replay_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_batch_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, game):
        self.epsilon = 0.001 + (max(500 - self.n_epochs, 0) * (0.9 - 0.001) / 500)
        u = random()
        random_action = u <= self.epsilon

        next_steps = game.get_next_states_tensor()
        # print(next_steps)
        next_actions, next_states = zip(*next_steps.items())
        # print(next_states)
        prediction = np.zeros(len(next_states))

        # self.model.eval()
        for i in range(len(next_states)):
            state = torch.tensor(next_states[i], dtype=torch.float)
            # print(state)
            with torch.no_grad():
                prediction[i] = self.model(state)
        # self.model.train()
        # print(prediction)
        prediction = torch.tensor(prediction, dtype=torch.float)
        # print(prediction)
        # print("Random action: ", random_action)

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(prediction).item()
        # print(index)

        return next_actions[index]

    def get_action_new(self, game, state):
        # self.epsilon = 0.001 + (max(500 - self.n_epochs, 0) * (0.9 - 0.001) / 500)
        self.epsilon = 80 - self.n_epochs
        u = randint(0, 200)
        random_action = u <= self.epsilon

        next_steps = game.get_next_states_tensor()
        next_actions, next_states = zip(*next_steps.items())
        # print(next_actions)
        print("Random action: ", random_action)

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            index = torch.argmax(prediction).item()

        return next_actions[index]
        # return index


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=24, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--max_epoch_score", type=int, default=100000, help="Maximum points per epoch")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--initial_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=0.001)
    parser.add_argument("--num_decay_epochs", type=float, default=500)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--replay_memory_size", type=int, default=700, help="Number of epochs between testing phases")
    parser.add_argument("--plot_scores", type=bool, default=True)
    parser.add_argument("--log_path", type=str, default="output/tensorboard")
    parser.add_argument("--saved_path", type=str, default="output/trained_models")

    args = parser.parse_args()
    return args


def train(options):
    plot_scores = []
    plot_average_scores = []
    total_score = 0
    max_score = 0
    agent = Agent()
    env = Tetris_new()
    while True:
        # get current state
        # states_old = agent.get_states(env)
        # print(states_old)
        state_old = agent.get_state(env)
        # print(state_old)
        # get move
        final_move = agent.get_action_new(env, state_old)
        # print(final_move)

        # perform move and get new state
        reward, score, game_over = env.step(final_move, render=RENDER)
        state_new = agent.get_state(env)

        # train short memory
        agent.train_batch_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        # print(reward)

        if game_over:
            print("Game Over ", final_move)
            # train replay memory, plotting
            env.reset()
            agent.n_epochs += 1
            agent.train_replay_memory()
            # c = b

            if score > max_score:
                max_score = score
                # agent.model.save()

            print('Game', agent.n_epochs, 'Score', score, 'Record', max_score)

            if PLOT:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_epochs
                plot_average_scores.append(mean_score)
                if agent.n_epochs % 1 == 0:
                    plot(plot_scores, plot_average_scores)


if __name__ == "__main__":
    opt = get_args()
    train(opt)
