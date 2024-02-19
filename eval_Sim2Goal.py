import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from config import *
import numpy as np
config = Config()
from data.loader import data_loader
from models.Sim2Goal import TrajectoryGenerator
from utils.utils import (
    relative_to_abs,
    get_dset_path,
    plot_best_goal,
    plot_best,
    fast_coll_counter
)

from utils.losses import displacement_error
from attrdict import AttrDict


seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)

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
        ids.append(id.squeeze().item() )
        sum_ += _error.squeeze()
    return sum_, ids


def get_generator(checkpoint):
    _args = AttrDict(checkpoint['args'])
    model = TrajectoryGenerator(_args)
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
    # final_error = (predicted_trajs[-1] - gt_traj[-1])**2
    return final_error.flatten()


def cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask):

    ade_col = displacement_error(pred_traj_fake, pred_traj_gt, val_mask, mode="raw")
    fde = compute_fde(pred_traj_fake.cpu().numpy(), pred_traj_gt.cpu().numpy(), val_mask[-1])
    ade = compute_ade(pred_traj_fake.cpu().numpy(), pred_traj_gt.cpu().numpy(), val_mask)
    return ade, fde, ade_col

def evaluate(args, loader, generator , gt_coll=False,
             plot_traj=False, plot_sample=False, robot = None):
    ade_outer, fde_outer = [], []
    total_traj = 0
    coll_pro_szenes_fake = 0.
    count_szenes_fake = 0.
    coll_pro_szenes = 0.
    count_szenes = 0.
    szene_id = 0.
    samples = None
    with torch.no_grad():
        for j, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, \
             loss_mask, seq_start_end, nei_num_index, nei_num = batch
            robots_ids = seq_start_end[:, 0]
            ade, fde, ade_col = [], [], []
            total_traj += nei_num.sum()
            batch_pred_traj_fake = []
            batch_samples = []
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            val_mask = val_mask[-config.pred_len :]

  
            sgoals = pred_traj_gt[-1] 

            pred_traj_fake_rel, _, _ = generator(model_input, obs_traj, pred_traj_gt,
                                            seq_start_end, nei_num_index, nei_num, plot_sample=plot_sample,
                                                    mode='test', sample_goal=sgoals,robot_net=robot,
                                                    robotID=robots_ids)

            pred_traj_fake_rel = pred_traj_fake_rel[-args.pred_len :]
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
           # pred_traj_fake[-1] = sgoals
            batch_samples.append(samples)
            batch_pred_traj_fake.append(pred_traj_fake)
            ade_, fde_, ade_col_ = cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask)
            if robot != None:
                ade_ = ade_[robots_ids.cpu().numpy()]
                fde_ = fde_[robots_ids.cpu().numpy()]
                ade_col_ = ade_col_[robots_ids.cpu().numpy()]
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
            szene_id += 1

        ade = np.mean(ade_outer) #/ (total_traj * args.pred_len)
        fde = np.mean(fde_outer) #/ (total_traj)
        act = coll_pro_szenes_fake / count_szenes_fake
        return ade, fde, act, 0

# check if trajnet is set to True in config if trajnet evaluation
# EXPERIMENT_NAME = 'SIM2Goal-ETHandUCY' # please change in config Trajnet =False
EXPERIMENT_NAME ='SIM2Goal-ETHandUCY'  # please change in config Trajnet =True
_dir = os.path.dirname(__file__)
_dir = _dir.split("/")[:-1]
_dir = "/".join(_dir) + "/Sim2Goal/models/weights/"
DIR = _dir + EXPERIMENT_NAME + '-' # -eth and so on
# /checkpoint_with_model.pt
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=DIR, type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def main(args):
    paths = []
    if config.trajnet:
        datasets = ['hotel', 'lcas', 'students1', 'students3', 'wildtrack', 'zara1', 'zara3']
    else:
        datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    for i in datasets:
        paths.append(args.model_path+i + '/checkpoint_with_model.pt')
    # paths = [args.model_path] + datasets
    for path in paths:
        checkpoint = torch.load(path)
        generator,sample_generator,sampler = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde, act, act_gt = evaluate(_args, loader, generator, sample_generator, sampler, gt_coll=False, plot_traj=False, plot_sample=False)
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
    main(args)
