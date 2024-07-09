import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


nn.Linear(16, 30)
class QNet(nn.Module):
    def __init__(self, MODEL_CONFIG):
        super().__init__()
        input_size = MODEL_CONFIG['input']
        self.hiddenLayer = MODEL_CONFIG['hiddenLayer']
        output_size = MODEL_CONFIG['output']
        for idx, layer in enumerate(self.hiddenLayer):
            if idx == 0:
                setattr(self, f'hidden{idx}', nn.Linear(input_size, layer['size']))
            else:
                setattr(self, f'hidden{idx}', nn.Linear(self.hiddenLayer[idx-1]['size'], layer['size']))
            # setattr(self, f'activation{idx}', getattr(F, layer['activation']))
        setattr(self, f'output', nn.Linear(self.hiddenLayer[-1]['size'], output_size))
        
    def forward(self, x):
        for idx in range(len(self.hiddenLayer)):
            # activation = getattr(self, f'activation{idx}')(x)
            x= F.relu(getattr(self, f'hidden{idx}')(x))
        x =  getattr(self, f'output')(x)
        return x
    


class TargetNetworkQTrainer:
    def __init__(self, model, model_target, lr, gamma, sfu, device):
        checkpoint  = torch.load('./model/training target model.pth')
        self.lr = lr
        self.device = device
        self.gamma = gamma
        self.model = model
        self.model.load_state_dict(checkpoint['model'])
        self.model_target = model_target
        self.model_target.load_state_dict(checkpoint['model_target'])
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer_target = optim.Adam(model_target.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.tau = sfu
        self.model.eval()
        self.model_target.eval()
    

    def soft_update(self, source_model, target_model, tau):
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            
    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)
        
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
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
    
    def save(self, file_name='training target model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({
            'model': self.model.state_dict(),
            'model_target': self.model_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_target': self.optimizer_target.state_dict(),
            'criterion': self.criterion.state_dict(),
            }, file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)
        
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
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
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

