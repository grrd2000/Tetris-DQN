import argparse
from random import random, randint, sample
import numpy as np
import torch
from src.model import DQNet, QTrainer
from src.tetris import Tetris
from src.tools import plot
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
GAMMA = 0.9
LR = 0.001
RENDER = False
PLOT = True


class Agent:
    batch_size = BATCH_SIZE
    gamma = GAMMA

    def __init__(self):
        self.n_epochs = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQNet(4, 128, 1)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_states(self, game):
        return game.get_next_states_tensor()

    def remember(self, state, reward, next_state, game_over):
        self.memory.append([state, reward, next_state, game_over])

    def train_replay_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        self.trainer.train_step(mini_sample)

    def train_batch_memory(self):
        self.trainer.train_step(self.memory)

    def get_action(self, game):
        self.epsilon = 0.001 + (max(450 - self.n_epochs, 0) * (0.925 - 0.001) / 450)
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

        return index

    def get_action_new(self, state):
        self.epsilon = 0.001 + (max(500 - self.n_epochs, 0) * (0.95 - 0.001) / 500)
        # self.epsilon = 90 - self.n_epochs
        u = random()
        random_action = u <= self.epsilon
        # print("Epsilon: ", self.epsilon)
        print("Random action: ", random_action)

        next_actions, next_states = zip(*state.items())

        next_states = torch.stack(next_states)
        # print(next_states, len(next_states))
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(next_states)[:, 0]
        self.model.train()

        if random_action:
            index = randint(0, len(state) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        return next_state, action


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
    parser.add_argument("--saved_path", type=str,
                        default="C:/Users/gerar/Documents/Projects/Programming/Tetris_DQN/output/trained_models")

    args = parser.parse_args()
    return args


def train(options):
    plot_scores = []
    plot_average_scores = []
    total_score = 0
    max_score = 0
    agent = Agent()
    env = Tetris()
    state = env.reset()
    while True:
        state_old = agent.get_states(env)

        final_move = agent.get_action_new(state_old)
        next_state, action = final_move

        reward, score, game_over = env.step(action, render=RENDER)

        agent.remember(state, reward, next_state, game_over)
        agent.train_batch_memory()

        if game_over:
            print("Game Over ", final_move)
            state = env.reset()
            agent.n_epochs += 1

            if score > max_score:
                max_score = score
                print("~BEST SCORE!~ Score: {}, Epoch: {}".format(max_score, agent.n_epochs))
                torch.save(agent.model, "{}/tetris_best_{}".format(options.saved_path, agent.n_epochs))

            print('Game', agent.n_epochs, 'Score', score, 'Record', max_score)

            if PLOT:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_epochs
                plot_average_scores.append(mean_score)
                if agent.n_epochs % 1 == 0:
                    plot(plot_scores, plot_average_scores)
        else:
            state = next_state


if __name__ == "__main__":
    opt = get_args()
    train(opt)
