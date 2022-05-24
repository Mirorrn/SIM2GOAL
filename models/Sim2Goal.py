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

class TrajectoryGenerator(nn.Module):
    def __init__(self,config):
        super(TrajectoryGenerator, self).__init__()
        self.config = config
        self.obs_len= config.obs_len
        self.pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.goal_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        # self.goal_encoder_dist = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16*1, rela_embed_size)
        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)
        self.pl_net_correct = Pooling_net(h_dim=traj_lstm_hidden_size)
        self.mlp_corrector = nn.Linear(traj_lstm_hidden_size, 2)
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.pred_lstm_model =  nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2*2)
        self.dropout = nn.Dropout(p=0.05)
        # self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.inputLayer_encoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder.weight, std=0.2)

        nn.init.constant_(self.inputLayer_decoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_decoder.weight, std=0.2)

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
        rel_goal = sample_goal - obs_traj_pos[-1]

        nll = 0.
        nll_robot = 0.
        rel_goal = self.goal_encoder(rel_goal)
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
            if robot_net != None:
                ids_no_robot = torch.zeros([self.config.pred_len, seq_start_end[-1, 1], 2], dtype=torch.bool).cuda()
                ids_no_robot[i, robotID, :] = True
            if not detached:
                input_cat = torch.cat([lstm_state_context, output], dim=-1)
            else:
                input_cat = torch.cat([lstm_state_context.detach(), output.detach()], dim=-1)
            input_embedded = self.dropout(F.relu(self.inputLayer_decoder(input_cat)))
            lstm_state = self.pred_lstm_model(
                input_embedded, (pred_lstm_hidden, pred_lstm_c_t)
            )
            pred_lstm_hidden = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0,1)-corr)
            lstm_state_hidden = lstm_state[0]
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)
            concat_output = lstm_state_context + lstm_state_hidden + rel_goal #+ emb_distance_goal.detach()
            mu, scale = self.pred_hidden2pos(concat_output).chunk(2, 1)
            scale = torch.clamp(scale, min=-9, max=4)
            if robot_net != None and noise != None:
                z_robot = noise[i, robotID]
                z_robot = robot_net(torch.cat([concat_output[ids_no_robot[i, :, 0]], mu[ids_no_robot[i, :, 0]], scale[ids_no_robot[i, :, 0]],z_robot], dim=-1))
                noise = noise.masked_scatter(ids_no_robot, z_robot)
            if mode == 'test':
                if noise == None:
                    output_pred_sampled = mu # Most likely sample
                else:
                    output_pred_sampled = mu + torch.exp(scale)*noise[i]  # for eval with a specific noise
            elif mode == 'robot_train':
                output_pred_sampled = mu + torch.exp(scale) * noise[i]
                nll_robot = nll_robot + GaußNLL(mu, scale, output_pred_sampled)
            else:
                output_pred_sampled = mu + torch.exp(scale) * torch.randn_like(scale)
                nll = nll + GaußNLL(mu, scale, traj_rel[self.obs_len + i])
            curr_pos_abs_sampled = (curr_pos_abs + output_pred_sampled).detach()
            output_pred = self.get_corrected(nei_index[i], lstm_state_hidden, nei_num_index, curr_pos_abs_sampled,
                                             output_pred_sampled, batch)
            # output_pred = output_pred_sampled  # for no Sampling module
            if not detached:
                curr_pos_abs = (curr_pos_abs + output_pred)
            else:
                curr_pos_abs = (curr_pos_abs + output_pred).detach()
            if plot_sample:
                plot_samples = curr_pos_abs.unsqueeze(dim=1) + mu.unsqueeze(dim=1) + scale.unsqueeze(dim=1) * torch.randn(
                    [scale.shape[0], 20, scale.shape[1]]).to('cuda')
                samples.append(plot_samples)
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
                return torch.stack(pred_traj_rel), mu, logscale
        elif mode =='robot_train':
            return torch.stack(pred_traj_rel), mu, logscale, nll_robot, torch.stack(pred_traj_rel_sampled) # For Robot training
        else:
            return torch.stack(pred_traj_rel), nll, torch.stack(pred_traj_rel_sampled) # Student learning
