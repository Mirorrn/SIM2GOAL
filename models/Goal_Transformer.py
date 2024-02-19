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

class Decoder_TF(nn.Module):
    def __init__(self):
        super(Decoder_TF, self).__init__()
        self.d_model = 16 +2
        nhead = 2
        dropout = 0.0
        d_hid = 32
        nlayers = 1
        max_t_len = 200
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(self.d_model, nhead, d_hid, dropout,layer_norm_eps=0.001)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(2 * self.d_model, self.d_model)
        self.input_fc = nn.Linear(2, self.d_model)
        self.output_fc = nn.Linear(self.d_model, 4)

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

    def forward(self, x, c):
        self.tgt_mask  = self.generate_square_subsequent_mask(x.shape[0])
        x = self.input_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0], x.shape[1], self.d_model)
        x_pos = self.positional_encoding(x, 8)
        x = self.transformer_decoder(x_pos,c, tgt_mask = self.tgt_mask )
        mu, scale = self.output_fc(x.reshape(-1, x.shape[-1])).view(x.shape[0],
                                                                    x.shape[1],
                                                                    4).chunk(2, 2)
        return mu, scale


class TrajectoryGenerator(nn.Module):
    def __init__(self,config):
        super(TrajectoryGenerator, self).__init__()
        self.config = config
        obs_len= config.obs_len
        pred_len= config.pred_len
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hist_encoder = Hist_Encoder(obs_len)
        self.decoder = Decoder_TF()

        self.robot_params_dict = {}
        self.robot_params_dict['use_robot_model'] = True # Flag for robot param constrains usage
        self.robot_params_dict['max_speed'] = .7  # [m/s] # 0.5 locobot
        self.robot_params_dict['min_speed'] = -.1  # [m/s]
        self.robot_params_dict['max_yaw_rate'] = 1.0  # [rad/s]
        self.robot_params_dict['max_accel'] = .5  # [m/ss]
        self.robot_params_dict['max_delta_yaw_rate'] = 3.2  # [rad/ss]
        self.dt=0.4
        self.predictions_steps = 12

    def actionXYtoROT(self, actionXY, robot_state, dt):
        # robot_state state[v_x(m), v_y(m), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
        v, yaw = self.cart2pol(actionXY[:, 0], actionXY[:, 1])
        omega_t = (yaw - robot_state[:, 2].unsqueeze(1)) / dt
        v_t = v / dt
        return torch.cat([v_t, omega_t], dim=-1)

    def cart2pol(self, x, y):
        rho = torch.sqrt(x ** 2 + y ** 2).unsqueeze(dim=1)
        phi = torch.arctan2(y, x).unsqueeze(dim=1)
        return rho, phi

    def dynamic_window(self, robot_state, u, params_dict, dt):
        # robot_state state[v_x(m / s), v_y(m / s), yaw(rad), v(m / s), omega(rad / s)] x and y are displacments to last state
        # Dynamic window from robot specification
        batch = u.shape[0]
        theta_current = robot_state[:, 2].unsqueeze(1)
        v_t = robot_state[:, 3].unsqueeze(1)
        omega = robot_state[:, 4].unsqueeze(1)
        v_new = u[:, 0].unsqueeze(1)
        yaw_new = u[:, 1].unsqueeze(1)
        filler = torch.ones_like(robot_state[:, 4]).unsqueeze(1)

        Vs = [params_dict["min_speed"] * filler, params_dict["max_speed"] * filler,
              -params_dict["max_yaw_rate"] * filler, params_dict["max_yaw_rate"] * filler]
        # Dynamic window from motion model
        Vd = [v_t - params_dict["max_accel"] * dt,
              v_t + params_dict["max_accel"] * dt,
              omega - filler * params_dict["max_delta_yaw_rate"] * dt,
              omega + filler * params_dict["max_delta_yaw_rate"] * dt]

        v_min = torch.max(torch.cat([Vs[0], Vd[0]], dim=1), dim=1)[0].unsqueeze(dim=1)
        v_max = torch.min(torch.cat([Vs[1], Vd[1]], dim=1), dim=1)[0].unsqueeze(dim=1)
        yaw_rate_min = torch.max(torch.cat([Vs[2], Vd[2]], dim=1), dim=1)[0].unsqueeze(dim=1)
        yaw_rate_max = torch.min(torch.cat([Vs[3], Vd[3]], dim=1), dim=1)[0].unsqueeze(dim=1)
        dw = [v_min, v_max, yaw_rate_min, yaw_rate_max]

        v_new = torch.clamp(v_new, min=dw[0], max=dw[1])
        yaw_new = torch.clamp(yaw_new, min=dw[2], max=dw[3])
        theta = (yaw_new * dt + theta_current)
        v_x, v_y = pol2cart(v_new, theta)
        new_robot_state = torch.cat([v_x, v_y, theta, v_new, yaw_new], dim=-1)
        return new_robot_state

    def get_robot_state(self, robot_traj_rel, dt=0.4):
        start_time = 7
        _, last_yaw = self.cart2pol(robot_traj_rel[start_time-1, 0],
                               robot_traj_rel[start_time-1, 1])
        v, yaw = self.cart2pol(robot_traj_rel[start_time, 0],
                            robot_traj_rel[start_time, 1])
        v_t = v / dt
        omega_t = (yaw - last_yaw) / dt
        if self.robot_params_dict['use_robot_model']:
            v_t = np.clip(v_t, a_min=self.robot_params_dict['min_speed'],
                              a_max=self.robot_params_dict['max_speed'])
            omega_t = np.clip(omega_t, a_min=-self.robot_params_dict['max_yaw_rate'],
                                  a_max=self.robot_params_dict['max_yaw_rate'])
            v_x, v_y = self.pol2cart(v_t, omega_t)

    def forward(self, traj_rel, obs_traj_pos, pred_traj_gt_pos, seq_start_end,
                nei_index, nei_num_index, mode="test", plot_sample=False,
                sample_goal=None, noise=None, robot_net= None, robotID= None, detached=True):
        batch = traj_rel.shape[1]
        nll = 0.
        # rel_goal = sample_goal - obs_traj_pos[-1]
        rel_goal = (sample_goal - obs_traj_pos[-1]).unsqueeze(dim=0).repeat(8, 1, 1)
        enc_hist = self.hist_encoder(traj_rel[: self.obs_len])
        noise_sampled = torch.randn(12, batch, 2).cuda()
        cond = torch.cat([rel_goal, enc_hist], dim = -1)
        mu, scale = self.decoder(noise_sampled, cond)
        scale = torch.clamp(scale, min=-9, max=4)
        if mode == 'robot_train':
            noise_sampled[robotID] = noise
        output_pred_sampled = mu + torch.exp(scale) * noise_sampled
        if mode != 'test':
            nll = GaußNLL(mu, scale, traj_rel[self.obs_len::])
        robot_state = self.get_robot_state(traj_rel)
        for i in range(self.predictions_steps):
            u = self.actionXYtoROT(output_pred_sampled[i, :], robot_state, self.dt)
            robot_state = self.dynamic_window(robot_state, u,
                                         self.robot_params_dict,
                                         self.dt)
            output_pred_sampled[i, :] = robot_state[:, :2] * self.dt

        return output_pred_sampled, nll, output_pred_sampled
