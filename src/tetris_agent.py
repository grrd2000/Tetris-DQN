import argparse
import os
import shutil
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from src.model import DeepQNetwork
from src.tetris import Tetris
from src.tools import plot
from collections import deque

video = False
scores = False
plots = True


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
    '''if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)'''
    torch.manual_seed(777)
    if os.path.isdir(options.log_path):
        shutil.rmtree(options.log_path)
    os.makedirs(options.log_path)
    writer = SummaryWriter(options.log_path)
    env = Tetris(width=options.width, height=options.height,
                 block_size=options.block_size, maxScore=options.max_epoch_score)

    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    criterion = nn.MSELoss()

    state = env.reset()

    replay_memory = deque(maxlen=options.replay_memory_size)
    epoch = 0
    best_score = 0
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    while epoch < options.num_epochs:
        next_steps = env.get_next_states_tensor()
        epsilon = options.final_epsilon + (max(options.num_decay_epochs - epoch, 0) * (
                options.initial_epsilon - options.final_epsilon) / options.num_decay_epochs)
        # print(epsilon)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        # print("next states", next_states, len(next_states))
        # print("next actions", next_actions, len(next_actions))
        next_states = torch.stack(next_states)
        # print("next states", next_states, len(next_states))
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            # print(predictions)
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, score, done = env.step(action, render=video)

        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
        else:
            state = next_state
            continue
        if len(replay_memory) < options.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), options.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + options.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        if scores:
            print("Epoch: {}/{},\t Tetromino: {},\t Action: {},\t Score: {},\t Tetrominoes {},\t Cleared lines: {}".format(
                epoch,
                options.num_epochs,
                env.piece,
                action,
                final_score,
                final_tetrominoes,
                final_cleared_lines))
            writer.add_scalar('Train/Score', final_score, epoch - 1)
            writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
            writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if options.plot_scores and plots:
            plot_scores.append(final_score)
            total_score += final_score
            mean_score = total_score / epoch
            plot_mean_scores.append(mean_score)
            if epoch % 1 == 0:
                plot(plot_scores, plot_mean_scores)

        if epoch > 0 and epoch % options.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(options.saved_path, epoch))
        if best_score < final_score and epoch >= 500:
            best_score = final_score
            print("~BEST SCORE!~ Score: {}, Epoch: {}".format(best_score, epoch))
            torch.save(model, "{}/tetris_best_{}".format(options.saved_path, epoch))
            torch.save(model, "{}/tetris_best".format(options.saved_path))

    torch.save(model, "{}/tetris".format(options.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
