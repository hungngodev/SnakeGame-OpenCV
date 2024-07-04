import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class TargetNetworkQTrainer:
    def __init__(self, model, model_target, lr, gamma, sfu):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.model_target = model_target
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.optimizer_target = optim.Adam(model_target.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.tau = sfu

    def soft_update(self, source_model, target_model, tau):
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            target[idx][torch.argmax(action[idx]).item()] = reward[idx] + self.gamma * torch.max(self.model_target(next_state[idx])) * (1 - done[idx])
    

        self.optimizer.zero_grad()
        
        loss = self.criterion(target, pred)
        #perform gradient
        loss.backward()
        
        #update weights of Q network 
        self.optimizer.step()

        self.soft_update(self.model, self.model_target, self.tau)


