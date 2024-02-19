import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from config import *
config = Config()
seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(0)
from utils.losses import GaußNLL
from models.utils import Pooling_net
# copy of Sim2Goal but without goal and different sampling during inference

class Hist_Encoder(nn.Module):
    def __init__(self, hist_len):
        super(Hist_Encoder, self).__init__()
        self.d_model = 16
        self.hist_len = hist_len
        nhead = 2
        dropout = 0.0
        d_hid = 32
        nlayers = 1
        max_t_len = 200
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.pred_hidden2pos = nn.Linear(16, 2 * 2)
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(2 * self.d_model, self.d_model)
        self.input_fc = nn.Linear(2, self.d_model)

        self.scr_mask = self.generate_square_subsequent_mask(self.hist_len)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=1)

        return pe

    def get_pos(self, num_t, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        # pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def positional_encoding(self, x, t_offset):
        num_t = x.shape[0]
        pos_enc = self.get_pos(num_t, t_offset)
        feat = [x, pos_enc.repeat(1, x.size(1), 1)]
        x = torch.cat(feat, dim=-1)
        x = self.fc(x)
        return self.dropout(x)


    def generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz, device='cuda') * float('-inf'), diagonal=1)

    def forward(self, x, c=None):
        # test = x.numpy()
        x = self.input_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0], x.shape[1], self.d_model)
        x_pos = self.positional_encoding(x, 0)
        x_enc = self.transformer_encoder(x_pos, mask=self.scr_mask)
        return x_enc

class TrajectoryGenerator(nn.Module):
    def __init__(self,config):
        super(TrajectoryGenerator, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16
        self.device='cuda'
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        # self.goal_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        # self.goal_encoder_dist = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)
        # self.mlp_corrector = nn.Linear(traj_lstm_hidden_size, 2)
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        # self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.hist_encoder = Hist_Encoder(obs_len)
        self.pred_lstm_model =  nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos1 = nn.Linear(traj_lstm_hidden_size +2 , traj_lstm_hidden_size)
        self.pred_hidden2pos2 = nn.Linear(traj_lstm_hidden_size, 2*2)
        self.dropout = nn.Dropout(p=0.0)

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, 16).cuda(),
            torch.randn(batch, 16).cuda(),
        )


    def forward(self, traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index, mode="test", plot_sample=False,
                sample_goal=None, noise=None, robot_net= None, robotID= None, detached=True):
        batch = traj_rel.shape[1]
        pred_traj_rel = []
        scale_sum = 0.
        nll = 0.

        rel_goal = sample_goal - obs_traj_pos[-1]
        enc_hist = self.hist_encoder(traj_rel[: self.obs_len])[-1]
        pred_lstm_hidden = enc_hist
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden, device=self.device)
        output = traj_rel[self.obs_len-1]
        lstm_state_context = torch.zeros_like(pred_lstm_hidden).cuda()
        curr_pos_abs = obs_traj_pos[-1]

        for i in range(self.pred_len):

            input_cat = torch.cat([lstm_state_context, output], dim=-1).detach()
            input_embedded = self.dropout(F.relu(self.inputLayer_decoder(input_cat)))
            lstm_state = self.pred_lstm_model(
                input_embedded, (pred_lstm_hidden, pred_lstm_c_t)
            )
            pred_lstm_hidden = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]

            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0, 1)-corr)

            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, pred_lstm_hidden, curr_pos_abs)
            concat_output = lstm_state_context + pred_lstm_hidden

            h = F.relu(self.pred_hidden2pos1(torch.cat([concat_output, rel_goal], dim=-1)))
            mu, scale = self.pred_hidden2pos2(h).chunk(2, 1)
            scale = torch.clamp(scale, min=-9, max=4)

            output_pred = mu + torch.exp(scale) * torch.randn_like(scale)
            nll = nll + GaußNLL(mu, scale, traj_rel[self.obs_len + i])
            curr_pos_abs = (curr_pos_abs + output_pred).detach()
            scale_sum = scale_sum + scale.sum()

            pred_traj_rel += [output_pred]
            output = output_pred



        return torch.stack(pred_traj_rel), nll, scale_sum
