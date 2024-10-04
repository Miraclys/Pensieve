import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import torch.optim as optim

GAMMA = 0.99
EPOCH = 100000

HIDDEN_DIM = 128
ACTOR_KERNEL_SIZE = 4

def get_entropy_weight(epoch):
    if epoch >= EPOCH:
        return 0.1
    else:
        return 1 - 0.9 * epoch / EPOCH

class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.fc1 = nn.Linear(1, HIDDEN_DIM)
        self.fc2 = nn.Linear(1, HIDDEN_DIM)
        self.conv1 = nn.Conv1d(1, HIDDEN_DIM, kernel_size=ACTOR_KERNEL_SIZE)
        self.conv2 = nn.Conv1d(1, HIDDEN_DIM, kernel_size=ACTOR_KERNEL_SIZE)
        self.conv3 = nn.Conv1d(1, HIDDEN_DIM, kernel_size=ACTOR_KERNEL_SIZE)
        self.fc3 = nn.Linear(1, HIDDEN_DIM)

        self.S_LEN = s_dim[1]
        self.S_INFO = s_dim[0]

        out_dim = (3 + 2 * ((self.S_LEN - ACTOR_KERNEL_SIZE) / 1 + 1) + \
                   ((self.a_dim - ACTOR_KERNEL_SIZE) / 1 + 1)) * HIDDEN_DIM

        # need to change
        # self.out_layer = nn.Linear(out_dim, self.a_dim)
        self.out_layer = nn.Linear(int(out_dim), int(self.a_dim))

    def forward(self, state):
        split_0 = F.relu(self.fc1(state[:, 0:1, -1]))
        split_1 = F.relu(self.fc2(state[:, 1:2, -1]))
        split_2 = F.relu(self.conv1(state[:, 2:3, :].view(-1, 1, self.S_LEN)))
        split_3 = F.relu(self.conv2(state[:, 3:4, :].view(-1, 1, self.S_LEN)))
        split_4 = F.relu(self.conv3(state[:, 4:5, :self.a_dim].view(-1, 1, self.a_dim)))
        split_5 = F.relu(self.fc3(state[:, 5:6, -1]))

        split_2_flatten, \
        split_3_flatten, \
        split_4_flatten = split_2.flatten(start_dim=1), \
                        split_3.flatten(start_dim=1), \
                        split_4.flatten(start_dim=1)
        
        output = torch.cat((split_0, split_1, split_2_flatten, 
                               split_3_flatten, split_4_flatten, split_5), dim=1)
        
        logits = self.out_layer(output)

        return logits
    
class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.fc1 = nn.Linear(1, HIDDEN_DIM)
        self.fc2 = nn.Linear(1, HIDDEN_DIM)
        self.conv1 = nn.Conv1d(1, HIDDEN_DIM, kernel_size=ACTOR_KERNEL_SIZE)
        self.conv2 = nn.Conv1d(1, HIDDEN_DIM, kernel_size=ACTOR_KERNEL_SIZE)
        self.conv3 = nn.Conv1d(1, HIDDEN_DIM, kernel_size=ACTOR_KERNEL_SIZE)
        self.fc3 = nn.Linear(1, HIDDEN_DIM)

        self.S_LEN = s_dim[1]
        self.S_INFO = s_dim[0]

        out_dim = (3 + 2 * ((self.S_LEN - ACTOR_KERNEL_SIZE) / 1 + 1) + \
                   ((self.a_dim - ACTOR_KERNEL_SIZE) / 1 + 1)) * HIDDEN_DIM

        self.out_layer = nn.Linear(int(out_dim), 1)

    def forward(self, state):
        split_0 = F.relu(self.fc1(state[:, 0:1, -1]))
        split_1 = F.relu(self.fc2(state[:, 1:2, -1]))
        split_2 = F.relu(self.conv1(state[:, 2:3, :].view(-1, 1, self.S_LEN)))
        split_3 = F.relu(self.conv2(state[:, 3:4, :].view(-1, 1, self.S_LEN)))
        split_4 = F.relu(self.conv3(state[:, 4:5, :self.a_dim].view(-1, 1, self.a_dim)))
        split_5 = F.relu(self.fc3(state[:, 5:6, -1]))

        split_2_flatten, \
        split_3_flatten, \
        split_4_flatten = split_2.flatten(start_dim=1), \
                        split_3.flatten(start_dim=1), \
                        split_4.flatten(start_dim=1)
        
        output = torch.cat((split_0, split_1, split_2_flatten, 
                               split_3_flatten, split_4_flatten, split_5), dim=1)
        
        value = self.out_layer(output)

        return value

if __name__ == '__main__':
    print((3 + 2 * ((S_LEN - ACTOR_KERNEL_SIZE) / 1 + 1) + \
           ((A_DIM - ACTOR_KERNEL_SIZE) / 1 + 1)) * HIDDEN_DIM)