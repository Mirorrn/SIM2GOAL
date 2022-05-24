import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
config = Config()
from models.utils import Pooling_net



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
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size + 16, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pl_net = Pooling_net(h_dim=traj_lstm_hidden_size)

        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size
        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size
        self.traj_lstm_model = nn.LSTMCell(rela_embed_size, 8)
        self.pred_lstm_model =  nn.LSTMCell(rela_embed_size, 16)
        self.pred_hidden2pos = nn.Linear(traj_lstm_hidden_size, 2)

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

        nn.init.constant_(self.pred_hidden2pos.bias, 0.0)
        nn.init.normal_(self.pred_hidden2pos.weight, std=0.1)

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, 8).cuda(),
            torch.randn(batch, 8).cuda(),
        )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)  # TODO: maybe not repeat!
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def forward(self, traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index, plot_att=False):
        batch = traj_rel.shape[1]
        pred_traj_rel = []
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        for i, input_t in enumerate(traj_rel[: self.obs_len].chunk(
                traj_rel[: self.obs_len].size(0), dim=0)):
            input_embedded = F.relu(self.inputLayer_encoder(input_t.squeeze(0)))
            lstm_state = self.traj_lstm_model(
                input_embedded, (traj_lstm_h_t, traj_lstm_c_t)
            )
            traj_lstm_h_t, traj_lstm_c_t = lstm_state

        pred_lstm_hidden = self.add_noise(traj_lstm_h_t, seq_start_end)
        pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
        output  = torch.zeros_like(traj_rel[self.obs_len - 1]).cuda()
        lstm_state_context = torch.zeros_like(pred_lstm_hidden).cuda()
        curr_pos_abs =  obs_traj_pos[-1]
        for i in range(self.pred_len):
            input_cat = torch.cat([lstm_state_context.detach(),output.detach()], dim=-1)
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
            output =  torch.tanh(self.pred_hidden2pos(concat_output))* 4.4
            curr_pos_abs = (curr_pos_abs + output).detach() # detach from history as input
            pred_traj_rel += [output]
        if plot_att:

            return torch.stack(pred_traj_rel), None
        else:
            return torch.stack(pred_traj_rel)

class Discriminator(nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        traj_lstm_input_size= 2
        traj_lstm_hidden_size=16
        rela_embed_size = 16
        self.inputLayer_encoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.inputLayer_decoder = nn.Linear(traj_lstm_input_size, rela_embed_size)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dropout = nn.Dropout(0.00)
        self.gcn = Pooling_net(h_dim=traj_lstm_hidden_size)
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size

        self.pred_lstm_hidden_size = self.traj_lstm_hidden_size

        self.pred_hidden2class = nn.Linear(64, 1)

        self.init_parameters()

        self.layer1 = nn.Linear(2*(config.pred_len + config.obs_len)+16 * 12, 32)
        self.layer2 = nn.Linear(32, 64)


    def init_parameters(self):
        nn.init.constant_(self.inputLayer_encoder.bias, 0.0)
        nn.init.normal_(self.inputLayer_encoder.weight, std=0.2)

        nn.init.constant_(self.pred_hidden2class.bias, 0.0)
        nn.init.normal_(self.pred_hidden2class.weight, std=0.2)

    def init_hidden_traj_lstm(self, batch, hidden_size):
        return (
            torch.randn(batch, hidden_size).cuda(),
            torch.randn(batch, hidden_size).cuda(),
        )

    def forward(self, obs_rel, traj_rel_pred, obs_traj_pos,
                nei_index, nei_num_index, loss_mask,plot_att=False):
        batch = obs_rel.shape[1]
        traj_rel_pred = traj_rel_pred * loss_mask
        curr_pos_abs = obs_traj_pos[-1].detach()
        inter_list = []
        for j, input_t_j in enumerate(traj_rel_pred):
            input_embedded = F.leaky_relu(self.inputLayer_decoder(input_t_j)) # detach from history as input
            corr = curr_pos_abs.repeat(batch, 1, 1)
            corr_index = (corr.transpose(0,1)-corr)
            s_context, _, _ = self.gcn(corr_index, nei_index[j], nei_num_index, input_embedded, curr_pos_abs)
            curr_pos_abs = (curr_pos_abs + input_t_j)
            inter_list.append(s_context)
        inter_list = torch.stack(inter_list, dim=1).view(-1, 16 * 12)
        rel_traj = torch.cat([obs_rel, traj_rel_pred])
        rel_traj = rel_traj.permute(1, 0, 2).reshape(batch, -1)

        x = torch.cat([rel_traj, inter_list], dim=-1)
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        cls = self.pred_hidden2class(x)

        return cls
