import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class DQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
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

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        print("state", state)
        print("action", action)
        print("next_state", next_state)
        print("reward", reward)
        print("game_over", game_over)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        print("pred", pred)

        # 2: Q_new = r + y * max(next_predicted Q value)
        # pred.clone()
        # preds[argmax(action)] = Q_new

        target = pred.clone()
        # print(reward)
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # print("idx", idx)
            # print("len", len(game_over))
            # print("game_over", game_over)
            # print("action", action)
            # print("len (action)", len(action))
            # print("action[idx]", action[idx])
            # print("argmax", torch.argmax(action))
            # print("item", torch.argmax(action).item())
            # print("target[][] shape", target.shape)
            # print("target", target)
            # print("target item", target[torch.argmax(action[idx])])
            # print("target argmax action", target[torch.argmax(action)])
            target[idx] = Q_new
            # print("target item", target[torch.argmax(action[idx])])
            # print("target argmax action", target[torch.argmax(action)])
            # target[idx][torch.argmax(action).item()] = Q_new
            # target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
