import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from config import *
import numpy as np
config = Config()
from data.loader import data_loader
from models.GoalFLow import GoalGenerator
from utils.utils import (
    get_dset_path,
    plot_goals,

)
from utils.losses import final_displacement_error
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
    model = GoalGenerator(config)
    model.load_state_dict(checkpoint["best_state"])
    model.cuda()
    model.eval()
    return model


def cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask):
    fde = final_displacement_error(pred_traj_fake, pred_traj_gt[-1], val_mask[-1], mode="raw")
    return fde



def evaluate(args, loader, generator, plot_traj=False, plot_sample=False):
    fde_outer = []
    total_traj = 0
    szene_id = 0.
    samples = None
    eval_fde_batch_errors = np.array([])
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, \
             loss_mask, seq_start_end, nei_num_index, nei_num = batch
            all_l2_errors_dest = []
            total_traj += nei_num.sum()
            batch_pred_traj_fake = []
            batch_samples = []
            att_score_list_batch = []
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            val_mask = val_mask[-config.pred_len :]
            for _ in range(args.num_samples):
                if plot_sample:
                    pred_traj_fake, samples = generator(model_input, obs_traj, pred_traj_gt,
                                                   seq_start_end, nei_num_index, nei_num, plot_sample=plot_sample)
                else:
                    pred_traj_fake = generator(model_input, obs_traj, pred_traj_gt,
                                                    seq_start_end, nei_num_index, nei_num)
                batch_samples.append(samples)
                batch_pred_traj_fake.append(pred_traj_fake)
                fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake, val_mask)
                all_l2_errors_dest.append(fde_.cpu().numpy())

                # fde_sum, ids = evaluate_helper(fde, seq_start_end)

            all_l2_errors_dest = np.array(all_l2_errors_dest)
            # all_guesses = np.array(all_guesses)
            # average error
            # l2error_avg_dest = np.mean(all_l2_errors_dest)

            # choosing the best guess
            # indices = np.argmin(all_l2_errors_dest, axis=0)

            # best_guess_dest = all_guesses[indices, np.arange(obs_traj.shape[0]), :]
            l2error_dest = np.min(all_l2_errors_dest, axis=0)
            eval_fde_batch_errors = np.hstack([eval_fde_batch_errors, l2error_dest])
            if (plot_traj):
                plot_goals(obs_traj, pred_traj_gt, batch_pred_traj_fake, seq_start_end, batch_samples)
            szene_id += 1

        fde = np.mean(eval_fde_batch_errors)  # / (total_traj)

        return fde


# check if trajnet is set to True in config if trajnet evaluation
# EXPERIMENT_NAME = 'GFLOW-ETHandUCY' # eth
EXPERIMENT_NAME = 'GFLOW-TrajNet' # eth
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
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        fde = evaluate(_args, loader, generator, plot_traj=False, plot_sample=False)
        print(
            "Dataset:  {} \n"
            "Pred Len: {} FDE \n"
            "{:.4f}".format(
                _args.dataset_name, _args.pred_len, fde
            )
        )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)