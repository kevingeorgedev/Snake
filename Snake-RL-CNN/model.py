import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import os
import numpy as np

from game import SnakeGameAI

class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_linear = nn.Sequential(
            nn.Linear(in_features=3072, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU()
        )

        self.linear = nn.Linear(in_features=15, out_features=128)
        #self.linear2 = nn.Linear(in_features=256, out_features=3)
        self.act = nn.ReLU()

        self.output = nn.Linear(in_features=256, out_features=3)

    def forward(self, state, board):
        x = self.conv1(board)
        x = nn.Flatten()( x )
        x = self.conv_linear(x)
        state = self.act(self.linear(state.float()))
        x = torch.cat((x, state), dim=1)
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
        #print("TRAIN STEP")
        #print(len(state), len(state[0]), len(state[1]))
        #if len(state) > 1:#
        #    print(len(state[0]))
        #print(np.array(state.to('')).ndim)
        #print(type(state[0]))
        #print(len(state))
        #pred = torch.empty()
        #state_ = torch.tensor(state[0], dtype=torch.float, device=self.device)#.unsqueeze(0)
        state_ = state[0]
        board = state[1]
        #next_state_ = torch.tensor(next_state[0], dtype=torch.float, device=self.device)
        next_state_ = next_state[0]
        next_board = next_state[1]
        action = torch.tensor(action, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        # (n, x)
        #print(board.shape, next_board.shape)
        #print(torch.is_tensor(state_))
        #if len(state_.shape) == 1:
        if torch.is_tensor(state_):
            # (1, x)
            #state_      = torch.unsqueeze(state_, 0)
            #board      = torch.unsqueeze(board, 0)
            next_state_ = torch.unsqueeze(next_state_, 0)
            #next_board = torch.unsqueeze(next_board, 0)
            action      = torch.unsqueeze(action, 0)
            reward      = torch.unsqueeze(reward, 0)
            done        = (done, )
            pred = self.model(state_, board)
            #print("first", pred.shape)
            target = pred.clone()

            #print(len(done))
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    #print("HERE2")
                    #print(len(next_state_))
                    #print(next_state_.shape)
                    #print(next_board.shape)
                    #print(len(next_state_), len(next_board))
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_[idx], next_board))

                target[idx][torch.argmax(action).item()] = Q_new
        else:
            pred = []
            for p in state:
                output = self.model(*p)
                pred.append(output)
            pred = torch.stack(pred).squeeze(dim=1)
            #print("second", pred.shape)

            target = pred.squeeze(dim=1).clone()
            #print(type(next_state))
            #print(len(done))
            #print(len())
            #print(len(pred))
            #print(len(done))
            #print(target.shape)
            #print(type(reward))
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    #print(len(next_state), len(next_state[idx]))
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx][0], next_state[idx][1]))
                    #print(next_state[idx][0].shape, next_state[idx][1].shape)
                target[idx][torch.argmax(action).item()] = Q_new

        #print(len(done))
        #print("HERE1")
        #print(len(state))
        # 1: predicted Q values with current state
        #for idx in range(len(state)):
        #if torch.is_tensor(state_):
            
        #else:
            

        

        # 2: Q_new = r + gamma * max( next_predicted Q value )
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
