import torch
import torch.nn as nn
import torch.optim as optim
import os
from random import sample
import numpy as np


class DQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = F.sigmoid(self.linear1(x))
        # x = self.linear2(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './trained_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, replay_memory):
        '''state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        print("state", state)
        print("pred", pred)

        # 2: Q_new = r + y * max(next_predicted Q value)
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            if len(game_over) > 1:
                target[idx] = Q_new
'''

        if len(replay_memory) <= 100_000 / 10:
            batch = sample(replay_memory, min(len(replay_memory), 1000))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = torch.stack(tuple(state for state in state_batch))
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = torch.stack(tuple(state for state in next_state_batch))

            q_values = self.model(state_batch)
            self.model.eval()
            with torch.no_grad():
                next_prediction_batch = self.model(next_state_batch)
            self.model.train()

            y_batch = torch.cat(
                tuple(reward if done else reward + self.gamma * prediction for reward, done, prediction in
                      zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, y_batch)
            loss.backward()

            self.optimizer.step()
