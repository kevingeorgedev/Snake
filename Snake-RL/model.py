import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.norm = nn.LayerNorm(hidden_size)

        #self.fc2 = nn.Linear(hidden_size, 128)
        #self.norm2 = nn.LayerNorm(128)

        #self.fc3 = nn.Linear(128, 32)
        #self.norm3 = nn.LayerNorm(32)
        
        self.output = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        """#x = self.norm(self.act(self.fc1(x)))
        
        #x = self.norm2(self.act(self.fc2(x)))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))"""
        x = self.act(self.fc1(x))
        #x = self.norm(self.act(self.fc1(x)))
        #x = self.norm2(self.act(self.fc2(x)))
        #x = self.norm3(self.act(self.fc3(x)))

        x = self.output(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model: Linear_QNet, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda:0')
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        # (n, x)

        #print(len(state.shape))
        if len(state.shape) == 1:
            # (1, x)
            state      = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action     = torch.unsqueeze(action, 0)
            reward     = torch.unsqueeze(reward, 0)
            done       = (done, )
        #print(state.shape)
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            #if Q_new > 0:
            #    print(Q_new)
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + gamma * max( next_predicted Q value )
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
