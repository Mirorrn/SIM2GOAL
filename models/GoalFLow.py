import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
config = Config()
import models.flows as fnn
from utils.losses import diversit_obj
from models.utils import Pooling_net

class GoalGenerator(nn.Module):
    def __init__(self,config):
        super(GoalGenerator, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)
        self.clamp_flow = config.clamp_flow

        modules = []
        for _ in range(4):
            modules += [
                # fnn.ActNorm(traj_lstm_input_size),
                fnn.MADE(traj_lstm_input_size, traj_lstm_hidden_size,
                         traj_lstm_hidden_size, act='relu',clamp=self.clamp_flow),
                fnn.Reverse(2)
            ]
        self.maf = fnn.FlowSequential(*modules)
        for module in self.maf.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        self.maf.to('cuda')
        # self.train()

        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.pred_lstm_model =  nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2)
        self.dropout = nn.Dropout(p=0.1)

        self.init_parameters()
        self.noise_dim = (8,)
        self.noise_type = 'gaussian'
        self.noise_mix_type = 'global'

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


    def forward(self, traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index, mode="test", plot_sample=False, sampling_module=None):
        batch = traj_rel.shape[1]

        nll = 0.
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        final_goal_gt = pred_traj_gt_pos[-1] - obs_traj_pos[-1]
        for i, input_t in enumerate(traj_rel[: self.obs_len].chunk(
                traj_rel[: self.obs_len].size(0), dim=0)):
            input_embedded = self.dropout(F.relu(self.inputLayer_encoder(input_t.squeeze(0))))
            lstm_state = self.traj_lstm_model(
                input_embedded, (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        pred_lstm_hidden = traj_lstm_h_t
        output = traj_rel[self.obs_len]
        curr_pos_abs = obs_traj_pos[-1]

        corr = curr_pos_abs.repeat(batch, 1, 1)
        corr_index = (corr.transpose(0,1)-corr)

        # lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, pred_lstm_hidden, curr_pos_abs)
        concat_output = pred_lstm_hidden #+ lstm_state_context
        noise = torch.randn([concat_output.shape[0],2])


        if mode == 'sampling':
            noise_sampled = sampling_module(concat_output.detach()).permute(1,0,2)
            output_pred_k_all = []
            for noise_sampled_k in noise_sampled:
                output_pred_k = self.maf.sample(concat_output.detach(), output.detach(), noise=noise_sampled_k)
                nll = nll - self.maf.log_probs(output_pred_k, concat_output.detach(), output.detach())[1]  # + l2
                output_pred_k_all.append(output_pred_k + obs_traj_pos[-1])
            output_pred_k = torch.stack(output_pred_k_all)
            div_loss = diversit_obj(output_pred_k, seq_start_end)
            nll = nll.mean() - div_loss # - div_loss to maximise the distanze!
        # output_pred = self.pred_hidden2pos(concat_output)  + obs_traj_pos[-1]
        elif mode != 'test':
            nll = nll - self.maf.log_probs(final_goal_gt, concat_output, output.detach())[1] #+ min_k_fde
        output_pred = self.maf.sample(concat_output, output.detach(), noise=noise) + obs_traj_pos[-1]
        if plot_sample:
            plt_s = []
            for l in range(20):
                noise = torch.randn([concat_output.shape[0], 2])
                plt_s.append(self.maf.sample(concat_output, output.detach(), noise=noise) + obs_traj_pos[-1])
            samples = torch.stack(plt_s)

        if mode == 'test':
            if plot_sample:
                return output_pred, samples
            else:
                return output_pred
        elif mode == 'train':
            return output_pred, nll.mean()
        elif mode == 'sampling':
            return output_pred_k, nll
