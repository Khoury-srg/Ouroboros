import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Subset

class FC(nn.Module):
    def __init__(self, num_hidden_layer, input_dim, hidden_dim, output_dim):
        super(FC, self).__init__()
        self.num_layer = num_hidden_layer+1
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_hidden_layer-1):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        for i in range(self.num_layer-1):
            x = F.relu(self.fc[i](x))
        return self.fc[-1](x)
