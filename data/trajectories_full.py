import logging
import os
import math
# from IPython import embed
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
from config import *
config = Config()
from data.augmenter import Transformer

class Collate():
    def __init__(self,augment, data_name):
        self.augment = augment
        self.data_name = data_name
    def seq_collate(self, data):
        (
            obs_seq_list,
            pred_seq_list,
            val_mask_list,
            loss_mask_list,
            loss_maks_rel,
        ) = zip(*data)

        _len = [len(seq) for seq in obs_seq_list]
        cum_start_idx = [0] + np.cumsum(_len).tolist()
        seq_start_end = [
            [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        seq_start_end = torch.LongTensor(seq_start_end)
        if self.augment and ('zara' in self.data_name):
            obs_seq_list, pred_seq_list = Transformer.transform_1([obs_seq_list, pred_seq_list])
        obs_seq_list = np.concatenate(obs_seq_list, axis=0)
        pred_seq_list = np.concatenate(pred_seq_list, axis=0)

        if self.augment and ('zara' not in self.data_name):
            obs_seq_list, pred_seq_list = Transformer.transform_2([obs_seq_list, pred_seq_list],seq_start_end)
        hole_obs = np.concatenate([obs_seq_list, pred_seq_list], axis=-1)
        loss_mask = np.concatenate(loss_mask_list, axis=0)
        val_mask = np.concatenate(val_mask_list, axis=0)
        loss_maks_rel = np.concatenate(loss_maks_rel, axis=0)


        if config.nabs:
            rel_hole_obs = hole_obs - obs_seq_list[:, :,-1]
            rel_hole_obs = rel_hole_obs*loss_mask #
        else:
            rel_hole_obs = np.zeros(hole_obs.shape)
            rel_hole_obs[:, :, 1:] = hole_obs[:, :, 1:] - hole_obs[:, :, :-1]
            rel_hole_obs = rel_hole_obs*loss_maks_rel # zero padded rel filtering and first pos

        # Data format: batch, input_size, seq_len
        # LSTM input format: seq_len, batch, input_size
        obs_traj_rel = torch.from_numpy(rel_hole_obs[:, :, : config.obs_len]).type(torch.float).permute(2, 0, 1)
        pred_traj_rel = torch.from_numpy(rel_hole_obs[:, :, config.obs_len :]).type(torch.float).permute(2, 0, 1)
        obs_traj = torch.from_numpy(obs_seq_list).type(torch.float).permute(2, 0, 1)
        pred_traj = torch.from_numpy(pred_seq_list).type(torch.float).permute(2, 0, 1)
        loss_mask = torch.from_numpy(loss_mask).type(torch.float).permute(2, 0, 1)
        val_mask = torch.from_numpy(val_mask).type(torch.float).permute(2, 0, 1)




        nei_num_index = torch.zeros([config.pred_len ,obs_traj.shape[1], obs_traj.shape[1]])
        nei_num = torch.zeros([len(seq_start_end)])

        for j in range(config.pred_len):
            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                V = loss_mask[j+config.obs_len, start:end,0] # care first pose is zero!
                A = torch.ger(V,V)
                nei_num_index[j, start:end, start:end] = A
                nei_num[i] = val_mask[-1, start:end,0].sum()
            nei_num_index[j] = nei_num_index[j].fill_diagonal_(0).float()

        out = [
            obs_traj,
            pred_traj,
            obs_traj_rel,
            pred_traj_rel,
            val_mask,
            loss_mask,
            seq_start_end,
            nei_num_index,
            nei_num,
            # nei_num_index_val
        ]

        return tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self,
            data_dir,
            obs_len=8,
            pred_len=12,
            skip=1,
            threshold=0.002,
            min_ped=1,
            delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        loss_mask_rel = []
        val_loss_mask = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # curr_seq_data is a 20 length sequence
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                loss_mask_tmp = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                rel_loss_mask = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                val_mask = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                num_partial_ped = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        if config.all_rel_persons:
                            con1 = (obs_len - (pad_front+1)) # we accept only traj which have at least a length in obs
                            con2 = pad_end - obs_len        # and in prediction as well
                            assert (con1 + con2) == (pad_end - pad_front)-1
                            if not (con1 >= 1 and con2 >= 1):
                                continue
                        else:
                            continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq[0,:] -= config.min_x
                    curr_ped_seq[1,:] -= config.min_y

                    _idx = num_partial_ped
                    if pad_end - pad_front == self.seq_len:
                        val_mask[_idx, :, :] = 1.
                        num_peds_considered += 1
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    rel_loss_mask[_idx, :, pad_front+1:pad_end] = 1. # +1 because in rel_coords the first entry should be always zero
                    loss_mask_tmp[_idx, :, pad_front:pad_end] = 1
                    num_partial_ped += 1
                if num_peds_considered > min_ped:                                   # only multi traj no single, for single use >=
                    num_peds_in_seq.append(num_partial_ped)
                    loss_mask_list.append(loss_mask_tmp[:num_partial_ped])
                    loss_mask_rel.append(rel_loss_mask[:num_partial_ped])
                    seq_list.append(curr_seq[:num_partial_ped])
                    val_loss_mask.append(val_mask[:num_partial_ped])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask = np.concatenate(loss_mask_list, axis=0)
        self.loss_maks_rel = np.concatenate(loss_mask_rel, axis=0)
        val_loss_mask = np.concatenate(val_loss_mask, axis=0)
        # non_linear_ped = np.asarray(non_linear_ped)
        self.obs_traj = seq_list[:, :, : self.obs_len]
        self.pred_traj = seq_list[:, :, self.obs_len :]
        self.loss_mask = loss_mask
        self.val_mask = val_loss_mask
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        print('Data loaded!')

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.val_mask[start:end, :],
            self.loss_mask[start:end, :],
            self.loss_maks_rel[start:end, :],
        ]
        return out
