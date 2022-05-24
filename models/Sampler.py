import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
config = Config()

class Sampler(nn.Module):
    def __init__(self,config):
        super(Sampler, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16

        self.L1 = nn.Linear(traj_lstm_hidden_size, 32)
        self.L2 = nn.Linear(32, 32)
        self.L3 = nn.Linear(32, config.num_samples*2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, concat_output):
        # eps = torch.randn_like(concat_output).to(concat_output)
        x = F.relu(self.L1(concat_output))
        x = F.relu(self.L2(x))
        samples = torch.tanh(self.L3(x))*2
        samples = samples.view(concat_output.shape[0],config.num_samples,2)
        return samples

