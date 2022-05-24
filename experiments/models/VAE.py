import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.losses import GaußNLL
from models.utils import Pooling_net

class TrajectoryGenerator(nn.Module):
    def __init__(self,config):
        super(TrajectoryGenerator, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16
        latent_dim = self.latent_dim = self.config.latent_dim
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_encoder_fut = nn.Linear(traj_lstm_hidden_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 18, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)

        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 16)
        self.traj_lstm_model_future = nn.LSTMCell(rela_embed_size, traj_lstm_hidden_size)
        self.pred_lstm_model =  nn.LSTMCell(rela_embed_size, 16)

        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2*2)

        # latent encoder during training
        self.inputLayer_encoder1 = nn.Linear(traj_lstm_hidden_size * 2, latent_dim)
        self.inputLayer_encoder2 = nn.Linear(traj_lstm_hidden_size * 2, latent_dim)
        # latent encoder during testing
        self.inputLayer_encoder_test1 = nn.Linear(traj_lstm_hidden_size, latent_dim)
        self.inputLayer_encoder_test2 = nn.Linear(traj_lstm_hidden_size, latent_dim)
        # latent decoder layer
        self.latent_dec = nn.Linear(latent_dim, traj_lstm_hidden_size)

        self.init_parameters()
        self.noise_dim = (8,)
        self.noise_type = 'gaussian'
        self.noise_mix_type = 'global'

    def init_parameters(self):
        nn.init.constant_(self.inputLayer_encoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder.weight, std=0.02)

        nn.init.constant_(self.inputLayer_decoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_decoder.weight, std=0.02)

        nn.init.constant_(self.inputLayer_encoder_fut.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder_fut.weight, std=0.02)

        # LSTM past
        nn.init.xavier_uniform_(self.traj_lstm_model.weight_ih)
        nn.init.orthogonal_(self.traj_lstm_model.weight_hh, gain=0.001)
        nn.init.constant_(self.traj_lstm_model.bias_ih, 0.0)
        nn.init.constant_(self.traj_lstm_model.bias_hh, 0.0)
        n = self.traj_lstm_model.bias_ih.size(0)
        nn.init.constant_(self.traj_lstm_model.bias_ih[n // 4:n // 2], 1.0)
        #LSTM future
        nn.init.xavier_uniform_(self.traj_lstm_model_future.weight_ih)
        nn.init.orthogonal_(self.traj_lstm_model_future.weight_hh, gain=0.001)
        nn.init.constant_(self.traj_lstm_model_future.bias_ih, 0.0)
        nn.init.constant_(self.traj_lstm_model_future.bias_hh, 0.0)
        n = self.traj_lstm_model_future.bias_ih.size(0)
        nn.init.constant_(self.traj_lstm_model_future.bias_ih[n // 4:n // 2], 1.0)
        #LSTM decoder
        nn.init.xavier_uniform_(self.pred_lstm_model.weight_ih)
        nn.init.orthogonal_(self.pred_lstm_model.weight_hh, gain=0.001)
        nn.init.constant_(self.pred_lstm_model.bias_ih, 0.0)
        nn.init.constant_(self.pred_lstm_model.bias_hh, 0.0)
        n = self.pred_lstm_model.bias_ih.size(0)
        nn.init.constant_(self.pred_lstm_model.bias_ih[n // 4:n // 2], 1.0)

        nn.init.constant_(self.pred_hidden2pos.bias, 0.0)
        nn.init.normal_(self.pred_hidden2pos.weight, std=0.01)

        #latent layers
        #encoder
        nn.init.constant_(self.inputLayer_encoder1.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder1.weight, std=0.02)
        #nn.init.xavier_uniform_(self.inputLayer_encoder1.weight)

        nn.init.constant_(self.inputLayer_encoder2.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder2.weight, std=0.02)
        #nn.init.xavier_uniform_(self.inputLayer_encoder2.weight)

        nn.init.constant_(self.inputLayer_encoder_test1.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder_test1.weight, std=0.02)
        #nn.init.xavier_uniform_(self.inputLayer_encoder_test1.weight)

        nn.init.constant_(self.inputLayer_encoder_test2.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder_test2.weight, std=0.02)
        #nn.init.xavier_uniform_(self.inputLayer_encoder_test2.weight)

        #decoder
        nn.init.constant_(self.latent_dec.bias, 0.0)
        nn.init.normal_(self.latent_dec.weight, std=0.02)

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )
    def latent_encoder(self, input_t, inference):
        # Apply relu activation and MLP layer to the inputs
        if not inference:  # if training, input_t is concatenation of past and future
            z_mu = 4 * torch.tanh(self.inputLayer_encoder1(input_t))  # mean [-1 1], zero mean
            z_log_var = 0.5*torch.sigmoid(self.inputLayer_encoder2(input_t))  # Var [1.01 ...]

        else:  # if testing, input_t is only past, different nn.Linear layer for testing
            z_mu = 4 * torch.tanh(self.inputLayer_encoder_test1(input_t))  # mean [-1 1], zero mean
            z_log_var = 0.5*torch.sigmoid(self.inputLayer_encoder_test2(input_t))  # Var [1.01 ...]

        mu = torch.reshape(z_mu, (input_t.shape[0], self.latent_dim))
        logVar = torch.reshape(z_log_var, (input_t.shape[0], self.latent_dim))

        return mu, logVar

    def latent_decoder(self, z):
        # decode latent variables
        x = F.relu(self.latent_dec(z))
        if self.config.softmax_dec:
            x = F.softmax(x, dim=-1)
        return x

    def reparameterize(self, mu, logVar, inference):
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)  # noise
        z = mu + std * eps
        if inference:
            z = mu + std * eps

        return z

    def forward(self,traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index, mode="test"):
        batch = traj_rel.shape[1]
        pred_traj_rel = []
        pred_mu = []
        pred_scale = []
        samples = []
        if mode == "test":
            inference = True
        else:
            inference = False

        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        for i, input_t in enumerate(traj_rel[: self.obs_len].chunk(
                traj_rel[: self.obs_len].size(0), dim=0)):
            input_embedded = F.dropout(F.relu(self.inputLayer_encoder(input_t.squeeze(0))),
                                       p=self.config.cond_dropout)
            lstm_state = self.traj_lstm_model(
                input_embedded, (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        past_h_t = traj_lstm_h_t
        z_mu_obs = torch.zeros(batch, self.latent_dim)
        z_var_log_obs = torch.ones(batch, self.latent_dim)
        z_mu, z_var_log = None, None
        lstm_state_context = torch.zeros_like(past_h_t)

        if not inference:  # if training
            # embed future into LSTM
            for j, input_t_fut in enumerate(traj_rel[self.obs_len:].chunk(
                    traj_rel[self.obs_len:].size(0), dim=0)):
                input_embedded_fut = F.relu(self.inputLayer_encoder(input_t_fut.squeeze(0)))
                lstm_state_fut = self.traj_lstm_model_future(input_embedded_fut+lstm_state_context, (traj_lstm_h_t, traj_lstm_c_t))
                traj_lstm_h_t, traj_lstm_c_t = lstm_state_fut
                curr_pos_abs = traj_rel[-j]
                corr = curr_pos_abs.repeat(batch, 1, 1)
                corr_index = (corr.transpose(0, 1) - corr)
                state_context, _, _ = self.pl_net(corr_index, nei_index[-j], nei_num_index,
                                                       traj_lstm_h_t, curr_pos_abs)
                lstm_state_context = F.relu(self.inputLayer_encoder_fut(state_context))

            future_context = lstm_state_context + traj_lstm_h_t

            # during training: concatenate past and future
            input_encoder = torch.cat((past_h_t, future_context), -1)
            z_mu, z_var_log = self.latent_encoder(input_encoder, inference)
            z = self.reparameterize(z_mu, z_var_log, inference)

        input_encoder = past_h_t  # only use past for z_mu_obs and z_var_log_obs
        z_mu_obs, z_var_log_obs = self.latent_encoder(input_encoder, True)

        if inference:  # if testing
            z = self.reparameterize(z_mu_obs, z_var_log_obs, inference)


        dec_z = self.latent_decoder(z)

        pred_lstm_hidden = past_h_t + dec_z

        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        output = traj_rel[self.obs_len]
        mu = output
        scale = torch.zeros_like(mu)
        lstm_state_context = torch.zeros_like(pred_lstm_hidden).cuda()
        curr_pos_abs = obs_traj_pos[-1]
        nll = 0.

        for i in range(self.pred_len):

            input_cat = torch.cat([lstm_state_context.detach(),mu.detach(), scale.detach()], dim=-1)
            input_embedded = F.relu(self.inputLayer_decoder(input_cat)) # detach from history as input
            lstm_state = self.pred_lstm_model(
                input_embedded, (pred_lstm_hidden, pred_lstm_c_t)
            )
            pred_lstm_hidden = lstm_state[0]
            pred_lstm_c_t = lstm_state[1]
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0,1)-corr)
            lstm_state_hidden = lstm_state[0]
            lstm_state_context, _, _ = self.pl_net(corr_index, nei_index[i], nei_num_index, lstm_state_hidden, curr_pos_abs)
            concat_output = lstm_state_context + lstm_state_hidden

            mu, scale = self.pred_hidden2pos(concat_output).chunk(2, 1)
            scale = torch.clamp(scale, min=-9, max=4)

            if mode == 'test':

                output_pred = mu + torch.exp(scale) * torch.randn_like(scale)

            else:
                output_pred = mu + torch.exp(scale) * torch.randn_like(scale)  #.to('cuda')
            nll = nll + GaußNLL(mu, scale, traj_rel[self.obs_len + i])

            curr_pos_abs = (curr_pos_abs + output_pred).detach() # detach from history as input
            pred_traj_rel += [output_pred]
            output = output_pred
            # pred_mu += [mu]
            # pred_scale += [scale]

        if mode=='test':
            return torch.stack(pred_traj_rel)
        else:
            return torch.stack(pred_traj_rel), nll, z_mu, z_var_log, z_mu_obs, z_var_log_obs
