import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import os

from game import SnakeGameAI

class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        """self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )"""

        self.linear = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(in_features=1228811, out_features=3)
        )

    def forward(self, states):
        state = states[0]
        img = states[1]
        print("Shape:", img.shape)
        img = img.permute(1,2,0)
        img = img.unsqueeze(0)
        x = self.conv1(img)
        #x = self.conv2(x)
        #print(states.shape, x.shape)
        #states = states.unsqueeze(0).repeat(states.size(0), 1)
        x = nn.Flatten()( x )
        #states = torch.flatten(states,)
        #states = states.unsqueeze(0)
        #if states.shape[0] > 1:
        #    x = x.repeat(states.shape[0], 1)
        print(state.shape, x.shape)
        #states = states.unsqueeze(-1).unsqueeze(-1)
        #x = torch.cat((x, states), dim=1)
        x = torch.cat((x, state), dim=1)
        x = self.linear(x)
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
        #print("TRAIN STEP")
        print(len(state), len(state[0]))
        state = torch.tensor(state[0], dtype=torch.float, device=self.device)#.unsqueeze(0)
        next_state = torch.tensor(next_state[0], dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state      = torch.unsqueeze(state, 0)
            #board      = torch.unsqueeze(board, 0)
            next_state = torch.unsqueeze(next_state, 0)
            #next_board = torch.unsqueeze(next_board, 0)
            action     = torch.unsqueeze(action, 0)
            reward     = torch.unsqueeze(reward, 0)
            done       = (done, )

        #print(len(done))
        #print("HERE1")
        #print(len(state))
        # 1: predicted Q values with current state
        #for idx in range(len(state)):

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                #print("HERE2")
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + gamma * max( next_predicted Q value )
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
