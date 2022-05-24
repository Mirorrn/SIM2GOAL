import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = True

from config import *
import numpy as np
config = Config()
from data.loader import data_loader
from models.SIMnoGoal import TrajectoryGenerator
from utils.utils import (
    relative_to_abs,
    get_dset_path,
    plot_best
)
from utils.losses import displacement_error
from attrdict import AttrDict

# Change to your path here!
DIR = home +'/Documents/NFTraj/SIMnoGoal_TrajNet-lcas/checkpoint_with_model.pt'


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=DIR, type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)
# import locale
# locale.setlocale(locale.LC_ALL, 'de_DE.utf8')

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    ids = []
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0).unsqueeze(dim=1)
        _error, id = torch.min(_error, 0)
        ids.append(id.squeeze().item())
        sum_ += _error.squeeze()
    return sum_, ids


def get_generator(checkpoint):
    model = TrajectoryGenerator(config)
    model.load_state_dict(checkpoint["best_state"])
    model.cuda()
    model.eval()
    return model

def compute_ade(predicted_trajs, gt_traj, val_mask):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=0)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj, val_mask):
    final_error = np.linalg.norm(predicted_trajs[-1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask):

    ade_col = displacement_error(pred_traj_fake, pred_traj_gt, val_mask, mode="raw")
    fde = compute_fde(pred_traj_fake.cpu().numpy(), pred_traj_gt.cpu().numpy(), val_mask[-1])
    ade = compute_ade(pred_traj_fake.cpu().numpy(), pred_traj_gt.cpu().numpy(), val_mask)
    return ade, fde, ade_col

def fast_coll_counter(pred_batch,seq_start_end, ids, mask):
    ids_of_col_szenes = np.zeros([seq_start_end.shape[0]])
    coll_pro_szene = 0
    stack_of_coll_indeces = []
    pred_batch = torch.stack(pred_batch)
    if not ids:
        mask = mask.unsqueeze(dim=0).repeat(config.num_samples, 1, 1, 1)
        mask = mask.view(config.num_samples * 12, pred_batch.shape[2], pred_batch.shape[2])
        pred = pred_batch.view(config.num_samples * 12, pred_batch.shape[2], 2)
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        if ids:
            pred = pred_batch[ids[i]]
        currSzene = pred[:, start:end]
        dist_mat = torch.cdist(currSzene, currSzene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
        dist_mat_one_triu = torch.triu(dist_mat)
        dist_mat_one_triu = dist_mat_one_triu #* mask[:,start:end, start:end]
        filter_zeros = torch.logical_and(0. != dist_mat_one_triu,  dist_mat_one_triu <= config.collision_distance)
        filter_col_pos = torch.logical_and(0. != dist_mat,  dist_mat <= config.collision_distance)
        filter_zeros_sum=filter_zeros.sum().unsqueeze(dim=0).item()
        coll_pro_szene += filter_zeros_sum
        if filter_zeros_sum > 0.:
            ids_of_col_szenes[i] = 1
        stack_of_coll_indeces.append(filter_col_pos)
    count = len(seq_start_end) #* count_empty
    if not ids:
        count = count * config.num_samples
    return coll_pro_szene, count, ids_of_col_szenes, stack_of_coll_indeces

def evaluate(args, loader, generator, gt_coll=False, plot_traj=False, plot_sample=False):
    ade_outer, fde_outer = [], []
    total_traj = 0
    coll_pro_szenes_fake = 0.
    count_szenes_fake = 0.
    coll_pro_szenes = 0.
    count_szenes = 0.
    szene_id = 0.
    samples = None
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, \
             loss_mask, seq_start_end, nei_num_index, nei_num = batch
            ade, fde, ade_col = [], [], []
            total_traj += nei_num.sum()
            batch_pred_traj_fake = []
            batch_samples = []
            att_score_list_batch = []
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            val_mask = val_mask[-config.pred_len :]
            for _ in range(args.num_samples):
                if plot_sample:
                    pred_traj_fake_rel, samples = generator(model_input, obs_traj, pred_traj_gt,
                                                   seq_start_end, nei_num_index, nei_num, plot_sample=plot_sample)
                else:
                    pred_traj_fake_rel = generator(model_input, obs_traj, pred_traj_gt,
                                                    seq_start_end, nei_num_index, nei_num)
                pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                # pred_traj_fake = pred_traj_fake_rel
                batch_samples.append(samples)
                batch_pred_traj_fake.append(pred_traj_fake)
                ade_, fde_, ade_col_ = cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask)
                ade.append(ade_)
                fde.append(fde_)
                ade_col.append(ade_col_)

            _, ids = evaluate_helper(ade_col, seq_start_end)
            ade_sum = np.array(ade)
            fde_sum = np.array(fde)
            fde_sum = np.min(fde_sum, axis=0)
            fde_outer = np.hstack([fde_outer, fde_sum])
            ade_sum = np.min(ade_sum, axis=0)
            ade_outer = np.hstack([ade_outer, ade_sum])

            coll_pro_szene_fake, count_fake, ids_of_col_szenes_fake, stack_of_coll_indeces = fast_coll_counter(batch_pred_traj_fake,
                                                                                        seq_start_end, None, nei_num_index)
            coll_pro_szenes_fake += coll_pro_szene_fake
            count_szenes_fake += count_fake
            if(plot_traj):
                plot_best(obs_traj, pred_traj_gt, batch_pred_traj_fake, att_score_list_batch, ids,
                               stack_of_coll_indeces, seq_start_end,
                               ids_of_col_szenes_fake, loss_mask, config, szene_id, batch_samples)

                #
                # plot_multimodal(obs_traj, pred_traj_gt, batch_pred_traj_fake, att_score_list_batch, ids,
                #                stack_of_coll_indeces, seq_start_end,
                #                ids_of_col_szenes_fake, loss_mask, config, szene_id, batch_samples)
            szene_id += 1

        ade = np.mean(ade_outer) #/ (total_traj * args.pred_len)
        fde = np.mean(fde_outer) #/ (total_traj)
        act = coll_pro_szenes_fake / count_szenes_fake
        if gt_coll:
            act_gt = coll_pro_szenes / count_szenes
            return ade, fde, act, act_gt
        return ade, fde, act, 0


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]
    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)

        _, loader = data_loader(_args, path)
        ade, fde, act, act_gt = evaluate(_args, loader, generator, gt_coll=False, plot_traj=False, plot_sample=False)
        print(
            "Dataset:  {} \n"
            "Pred Len: {}, ADE, FDE, ACT,ACT_gt \n"
            "{:.4f} \n" 
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f}".format(
                _args.dataset_name, _args.pred_len, ade, fde, act, act_gt
            )
        )


if __name__ == "__main__":
    args = parser.parse_args()
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    main(args)
