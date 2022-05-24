import torch
import torch.nn as nn
import numpy as np


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
        self.N = corr_index.shape[0]
        hj_t = lstm_state.unsqueeze(0).expand(self.N, self.N, self.h_dim)
        hi_t = lstm_state.unsqueeze(1).expand(self.N, self.N, self.h_dim)
        nei_index_t = nei_index.view((-1))
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