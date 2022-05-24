import torch.nn as nn
import torch.nn.functional as F
from torch import tanh

class Robot(nn.Module):
    def __init__(self):
        super(Robot, self).__init__()
        self.mlp_robot1 = nn.Linear(16 + 6, 16)
        self.mlp_robot2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.mlp_robot1(x))
        return tanh(self.mlp_robot2(x))*2.
