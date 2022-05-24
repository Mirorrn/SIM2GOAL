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

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class Pooling_net(nn.Module):
    def __init__(
            self, embedding_dim=32, h_dim=32,
            activation='relu', batch_norm=False, dropout=0.0
    ):
        super(Pooling_net, self).__init__()
        self.h_dim = h_dim
        self.bottleneck_dim = h_dim
        self.embedding_dim = embedding_dim

        self.mlp_pre_dim = embedding_dim + h_dim * 2
        self.mlp_pre_pool_dims = [self.mlp_pre_dim, 64, self.bottleneck_dim]
        self.attn = nn.Linear(self.bottleneck_dim, 1)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            self.mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def forward(self, corr_index, nei_index, nei_num, lstm_state, curr_pos_abs, plot_att=False):
        # if nei_index.sum() == 0:
        #     print('yoyo')
        #     return torch.zeros_like(lstm_state), (0, 0, 0), 0
        self.N = corr_index.shape[0]
        hj_t = lstm_state.unsqueeze(0).expand(self.N, self.N, self.h_dim)
        hi_t = lstm_state.unsqueeze(1).expand(self.N, self.N, self.h_dim)
        nei_index_t = nei_index.reshape((-1))
        corr_t = corr_index.reshape((self.N * self.N, -1))
        r_t = self.spatial_embedding(corr_t[nei_index_t > 0])
        mlp_h_input = torch.cat((r_t, hj_t[nei_index > 0], hi_t[nei_index > 0]), 1)
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)
        # Message Passing
        H = torch.full((self.N * self.N, self.bottleneck_dim), -np.Inf, device=torch.device("cuda"),dtype=curr_pool_h.dtype)
        H[nei_index_t > 0] = curr_pool_h
        pool_h = H.view(self.N, self.N, -1).max(1)[0]
        pool_h[pool_h == -np.Inf] = 0.
        return pool_h, (0, 0, 0), 0

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).to('cuda')
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).to('cuda')
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class TrajectoryGenerator(nn.Module):
    def __init__(self,config):
        super(TrajectoryGenerator, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.goal_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.goal_encoder_dist = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16*2, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)
        self.pl_net_correct = Pooling_net(h_dim=traj_lstm_hidden_size)
        self.mlp_corrector = nn.Linear(traj_lstm_hidden_size, 2)

        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.pred_lstm_model =  nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2*2)
        self.dropout = nn.Dropout(p=0.0)


    def init_parameters(self):
        nn.init.constant_(self.inputLayer_encoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder.weight, std=self.config.std_in)

        nn.init.constant_(self.inputLayer_decoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_decoder.weight, std=self.config.std_in)

        nn.init.xavier_uniform_(self.traj_lstm_model.weight_ih)
        nn.init.orthogonal_(self.traj_lstm_model.weight_hh, gain=0.001)

        nn.init.constant_(self.traj_lstm_model.bias_ih, 0.0)
        nn.init.constant_(self.traj_lstm_model.bias_hh, 0.0)
        n = self.traj_lstm_model.bias_ih.size(0)
        nn.init.constant_(self.traj_lstm_model.bias_ih[n // 4:n // 2], 1.0)

        nn.init.xavier_uniform_(self.pred_lstm_model.weight_ih)
        nn.init.orthogonal_(self.pred_lstm_model.weight_hh, gain=0.001)

        nn.init.constant_(self.pred_lstm_model.bias_ih, 0.0)
        nn.init.constant_(self.pred_lstm_model.bias_hh, 0.0)
        n = self.pred_lstm_model.bias_ih.size(0)
        nn.init.constant_(self.pred_lstm_model.bias_ih[n // 4:n // 2], 1.0)

        # nn.init.constant_(self.pred_hidden2pos.bias, 0.0)
        # nn.init.normal_(self.pred_hidden2pos.weight, std=self.config.std_out)

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, 16).cuda(),
            torch.randn(batch, 16).cuda(),
        )

    def get_corrected(self, nei_index, lstm_state_hidden, nei_num_index, curr_pos_abs, output_pred_sampled, batch):
        corr = curr_pos_abs.repeat(batch, 1, 1)
        corr_index = (corr.transpose(0, 1) - corr)
        lstm_state_context, _, _ = self.pl_net_correct(corr_index,nei_index, nei_num_index, lstm_state_hidden, curr_pos_abs)
        corrected_output = self.mlp_corrector(lstm_state_context) + output_pred_sampled
        return corrected_output


    def forward(self, traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index, mode="test", plot_sample=False,
                sample_goal=None,noise=None, robot_net= None, robotID= None, detached=True):
        batch = traj_rel.shape[1]
        pred_traj_rel = []
        pred_mu = []
        pred_scale = []
        samples = []
        pred_traj_rel_sampled = []
        final_goal_gt = sample_goal - obs_traj_pos[-1]
        goal = sample_goal

        nll = 0.
        final_goal_gt = self.goal_encoder(final_goal_gt)
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        for i, input_t in enumerate(traj_rel[: self.obs_len].chunk(
                traj_rel[: self.obs_len].size(0), dim=0)):
            input_embedded =self.dropout(F.relu(self.inputLayer_encoder(input_t.squeeze(0))))
            lstm_state = self.traj_lstm_model(
                input_embedded, (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        pred_lstm_hidden = traj_lstm_h_t
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        output = traj_rel[self.obs_len-1]
        lstm_state_context = torch.zeros_like(pred_lstm_hidden).cuda()
        curr_pos_abs = obs_traj_pos[-1]

        for i in range(self.pred_len):
            distance_goal = goal - curr_pos_abs
            emb_distance_goal = self.goal_encoder_dist(distance_goal)
            input_cat = torch.cat([lstm_state_context.detach(),output.detach(),emb_distance_goal], dim=-1)
            input_embedded = self.dropout(F.relu(self.inputLayer_decoder(input_cat))) # detach from history as input
            lstm_state = self.pred_lstm_model(
                input_embedded, (pred_lstm_hidden, pred_lstm_c_t)
            )
            pred_lstm_hidden = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0,1)-corr)
            lstm_state_hidden = lstm_state[0]
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)
            concat_output = lstm_state_context + lstm_state_hidden + final_goal_gt + emb_distance_goal.detach()
            mu, scale = self.pred_hidden2pos(concat_output).chunk(2, 1)
            scale = torch.clamp(scale, min=-9, max=4)
            nll = nll + GaußNLL(mu, scale, traj_rel[self.obs_len + i])
            if mode == 'test':
                if noise == None:
                    output_pred_sampled = mu #+ torch.exp(scale) #* torch.randn_like(scale) #+ output.detach() #.to('cuda')
                else:
                    output_pred_sampled = mu + torch.exp(scale)*noise[i]  # + torch.exp(scale)
            else:
                output_pred_sampled = mu + torch.exp(scale) * torch.randn_like(scale) #+ output.detach() #.to('cuda')
            curr_pos_abs_sampled = (curr_pos_abs + output_pred_sampled).detach()  # detach from history as input
            output_pred = self.get_corrected(nei_index[i], lstm_state_hidden, nei_num_index, curr_pos_abs_sampled,
                                             output_pred_sampled, batch)
            curr_pos_abs = (curr_pos_abs + output_pred).detach()  # detach from history as input
            pred_traj_rel += [output_pred]
            output = output_pred
            pred_mu += [mu]
            pred_scale += [scale]
            pred_traj_rel_sampled += [output_pred_sampled]

        mu = torch.stack(pred_mu)
        logscale = torch.stack(pred_scale)

        if mode == 'test':
            if plot_sample:
                return torch.stack(pred_traj_rel), torch.stack(samples)
            else:
                return torch.stack(pred_traj_rel), nll, torch.stack(pred_traj_rel_sampled)
        else:
            return torch.stack(pred_traj_rel),nll, torch.stack(pred_traj_rel_sampled)
