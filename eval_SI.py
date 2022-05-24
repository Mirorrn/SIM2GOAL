import argparse
import os

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from config import *
import numpy as np
config = Config()
from data.loader import data_loader
from models.Sim2Goal import TrajectoryGenerator
from models.Robot import Robot
from utils.utils import (
    relative_to_abs,
    get_dset_path,
    fast_coll_counter
)
from utils.losses import KL_gaussians
from attrdict import AttrDict


def get_generator(checkpoint, exp_name):
    _args = AttrDict(checkpoint['args'])
    model = TrajectoryGenerator(config)
    model.load_state_dict(checkpoint["best_state"])
    model.cuda()
    model.eval()
    checkpoint_robot_path = config.DIR + exp_name + '/checkpoint_with_model.pt' # from paper no SI
    checkpoint_sampler = torch.load(checkpoint_robot_path)
    robot = Robot()
    robot.load_state_dict(checkpoint_sampler["best_state"])
    robot.cuda().eval()

    return model,robot

def compute_ade(predicted_trajs, gt_traj, axis = 0):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=axis)
    if axis == 0:
        return ade.flatten()
    else:
        return ade

def compute_fde(predicted_trajs, gt_traj):
    final_error = torch.sqrt(((predicted_trajs-gt_traj)**2).sum(dim=-1))
    return final_error.mean(dim=0)

def evaluate_social(args, loader, Student, robot_net, plot_traj=False, sample_z=2, same_goal = False):
    mutual_info, ADE_min, ADE_max, MPE_vel_min,\
    MPE_vel_max, sum_timestep_KL_list, FDE,\
    ADE_avg_scene, MPE_avg_scene, MRE  = [], [], [], [], [], [], [], [], [], []
    coll_all, count_fake_all = 0., 0.
    sample = 20
    sample_z = sample_z
    same_goal = same_goal
    id_influencer = 0 # on Trajnet++ always zero
    szene_id = 0
    with torch.no_grad():
        for j,batch in enumerate(tqdm(loader)):
            if j>=100:
                break
            batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, \
             loss_mask, seq_start_end, nei_num_index, nei_num = batch
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            batch_size = seq_start_end.shape[0]
            robots_ids = seq_start_end[:,0]
            sgoals = pred_traj_gt[-1]
            current_pos = obs_traj[-1]
            dist_mat_pred = torch.cdist(sgoals, current_pos, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
            if same_goal == True:
                for _, (start, end) in enumerate(seq_start_end):
                   start = start.item()
                   end = end.item()
                   dist_mat_pred_scene = dist_mat_pred[start:end,start]
                   dist_mat_pred_scene[0] = np.inf
                   min_index = torch.argmin(dist_mat_pred_scene)
                   sgoals[start] = sgoals[min_index + start]

            #sgoals[robots_ids] = sgoals[seq_start_end[:,1]-1]
            sum_timestep_KL_batch, ADE_avg_scene_tmp_batch, MPE_avg_scene_tmp_batch,\
            MRE_batch, ADE_max_scene_batch, ADE_min_scene_batch,\
            MPE_vel_min_scene_batch, MPE_vel_max_scene_batch= [], [], [], [], [], [], [], []
            for l in range(sample_z):
                if l == 0:
                    z = torch.zeros_like(pred_traj_gt) # get the most likely behaviour of the humans
                else:
                    z = torch.randn_like(pred_traj_gt)
                conditioned_pos, conditioned_vel = [], []

                mu_c_samples = []
                logscale_c_samples = []

                for i in range(sample):
                    z_robot = torch.randn_like(pred_traj_gt)[:,0:batch_size]
                    z[:, robots_ids] = z_robot
                    conditioned, mu_c, logscale_c = Student(model_input, obs_traj, pred_traj_gt,
                                                              seq_start_end, nei_num_index, nei_num,
                                                              mode='test', sample_goal=sgoals, robot_net=robot_net,
                                                              robotID=robots_ids, noise=z)
                    pos_cond = relative_to_abs(conditioned, obs_traj[-1])
                    conditioned_pos.append(pos_cond)
                    conditioned_vel.append(conditioned)
                    mu_c_offset = torch.cat([obs_traj[-1].unsqueeze(dim=0), pos_cond], dim=0)
                    mu_c = mu_c + mu_c_offset[0:-1]
                    mu_c_samples.append(mu_c)
                    logscale_c_samples.append(logscale_c)

                coll_pro_szene_fake, count_fake, _, _ = fast_coll_counter(
                    conditioned_pos, seq_start_end, None, nei_num_index, sample=sample)
                conditioned_pos = torch.stack(conditioned_pos)
                coll_all += coll_pro_szene_fake
                count_fake_all += count_fake
                fde = compute_fde(conditioned_pos[:,-1], sgoals)[robots_ids]
                FDE = np.hstack([FDE, fde.cpu().numpy()])
                conditioned_vel = torch.stack(conditioned_vel) / 0.4
                # conditioned_acc = torch.zeros(conditioned_vel[:, :, robots_ids].shape) / 0.4
                # conditioned_acc[:, 1:] = conditioned_vel[:, 1:, robots_ids] - conditioned_vel[:, :-1, robots_ids]
                conditioned_acc = torch.zeros(conditioned_vel[:, :].shape)
                conditioned_acc[:, 1:] = (conditioned_vel[:, 1:] - conditioned_vel[:, :-1]) / 0.4
                conditioned_acc_robot = torch.sqrt((conditioned_acc[:, :, robots_ids]**2).sum(dim=-1)).mean(dim=1).cpu().numpy()
                mu_c_samples = torch.stack(mu_c_samples)
                logscale_c_samples = torch.stack(logscale_c_samples)
                ids_no_robot= torch.ones([seq_start_end[-1,1]], dtype=torch.bool).cuda()
                ids_no_robot[robots_ids] = False
                nei_num_index_tmp = nei_num_index[:, ids_no_robot]
                nei_num_index_tmp = nei_num_index_tmp[:, :, ids_no_robot]
                model_input_tmp = model_input[:, ids_no_robot]
                obs_traj_tmp = obs_traj[:, ids_no_robot]
                pred_traj_gt_tmp = pred_traj_gt[:, ids_no_robot]
                sgoals_tmp = sgoals[ids_no_robot]
                unconditioned_vel, mu_unc, logscale_unc = Student(model_input_tmp, obs_traj_tmp, pred_traj_gt_tmp,
                                                                  seq_start_end, nei_num_index_tmp, nei_num,
                                                                  mode='test', sample_goal=sgoals_tmp,
                                                                  noise=z[:, ids_no_robot])

                unconditioned_pos = relative_to_abs(unconditioned_vel, obs_traj_tmp[-1])
                unconditioned_vel = unconditioned_vel / 0.4
                unconditioned_acc = torch.zeros(unconditioned_vel.shape)
                unconditioned_acc[1:] = (unconditioned_vel[1:] - unconditioned_vel[:-1]) / 0.4
                mu_offset = torch.cat([obs_traj_tmp[-1].unsqueeze(dim=0),unconditioned_pos],dim = 0)
                mu_unc = mu_unc + mu_offset[0:-1]
                sum_timestep_KL = KL_gaussians(mu_c_samples[:,:, ids_no_robot], logscale_c_samples[:,:, ids_no_robot],
                                           mu_unc.unsqueeze(dim=0), logscale_unc.unsqueeze(dim=0), sum_dim=1)
                sum_timestep_KL_batch.append(sum_timestep_KL)

                seq_start_end_no_rob = seq_start_end.clone()
                f = torch.arange(0, batch_size).cuda()
                seq_start_end_no_rob[:, 0] = seq_start_end_no_rob[:, 0] - f
                seq_start_end_no_rob[:, 1] = seq_start_end_no_rob[:, 1] - f - 1
                ADE_max_scene, ADE_min_scene, MPE_vel_min_scene, MPE_vel_max_scene = [], [], [], []
                for (start, end),(start_r,end_r) in zip(seq_start_end,seq_start_end_no_rob):
                    if start_r == end_r:
                        continue
                    conditioned_pos_scene = conditioned_pos[:,:,start+1:end]
                    conditioned_vel_scene = conditioned_acc[:,:,start+1:end]
                    unconditioned_pos_scene = unconditioned_pos[:,start_r:end_r]
                    unconditioned_vel_scene = unconditioned_acc[:,start_r:end_r]
                    sum_batch = sum_timestep_KL[:,start_r:end_r]
                    sum_over_persons = sum_batch.sum(dim=-1)
                    max_KL = torch.max(sum_timestep_KL[:, start_r:end_r])
                    max_id = torch.argmax(sum_over_persons).cpu().numpy()
                    min_id = torch.argmin(sum_over_persons).cpu().numpy()
                    conditioned_pos_scene_min = conditioned_pos_scene[min_id]
                    conditioned_pos_scene_max = conditioned_pos_scene[max_id]
                    conditioned_vel_scene_min = conditioned_vel_scene[min_id]
                    conditioned_vel_scene_max = conditioned_vel_scene[max_id]

                    ADE_min_scene.append(compute_ade(conditioned_pos_scene_min.cpu().numpy(),unconditioned_pos_scene.cpu().numpy()))
                    ADE_max_scene.append(compute_ade(conditioned_pos_scene_max.cpu().numpy(),unconditioned_pos_scene.cpu().numpy()))
                    MPE_vel_min_scene.append(compute_ade(conditioned_vel_scene_min.cpu().numpy(),unconditioned_vel_scene.cpu().numpy()))
                    MPE_vel_max_scene.append(compute_ade(conditioned_vel_scene_max.cpu().numpy(),unconditioned_vel_scene.cpu().numpy()))
                   # if plot_traj and max_KL > 200:
                   #  plot_influencer_for_paper(obs_traj[:,start:end].cpu().numpy(), pred_traj_gt[:,start:end].cpu().numpy(),
                   #                   conditioned_pos[:,:,start:end].cpu().numpy(),
                   #                   id_influencer, ids_no_robot[start:end], unconditioned_pos[:,start_r:end_r].cpu().numpy(),
                   #                   sgoals[start:end].cpu().numpy(), min_id, szene_id)
                    szene_id += 1
                ADE_min_scene_batch.append(np.hstack(ADE_min_scene))
                ADE_max_scene_batch.append(np.hstack(ADE_max_scene))
                MPE_vel_min_scene_batch.append(np.hstack(MPE_vel_min_scene))
                MPE_vel_max_scene_batch.append(np.hstack(MPE_vel_max_scene))

                sum_timestep_KL_list = np.hstack([sum_timestep_KL_list, sum_timestep_KL.flatten().cpu().numpy()])


                ADE_avg_scene_tmp = compute_ade(conditioned_pos[:, :, ids_no_robot].cpu().numpy(),
                                                unconditioned_pos.cpu().numpy(), axis=1) # .mean(axis=0)
                ADE_avg_scene_tmp_batch.append(ADE_avg_scene_tmp)

                # con_test_acc = torch.zeros(conditioned_vel[:, :, ids_no_robot].shape)
                # test_vel = conditioned_vel
                # con_test_acc[:, 1:] = (test_vel[:, 1:, ids_no_robot] - test_vel[:, :-1, ids_no_robot])
                #
                # ucon_test_acc = torch.zeros(unconditioned_vel.shape)
                # utest_vel = unconditioned_vel
                # ucon_test_acc[:, 1:] = (utest_vel[:, 1:,] - utest_vel[:, :-1,])

                MPE_avg_scene_tmp = compute_ade(conditioned_acc[:, :, ids_no_robot].cpu().numpy(),
                                                unconditioned_acc.cpu().numpy(), axis=1) # .mean(axis=0)
                # MPE_avg_scene_tmp_test = compute_ade(con_test_acc.cpu().numpy(),ucon_test_acc.cpu().numpy(), axis=1) # .mean(axis=0)
                MPE_avg_scene_tmp_batch.append(MPE_avg_scene_tmp)
                MRE_batch.append(conditioned_acc_robot)

            ADE_min_scene_batch = np.stack(ADE_min_scene_batch)
            ADE_min = np.hstack([ADE_min, ADE_min_scene_batch.sum(axis=0) / (sample_z)])
            ADE_max_scene_batch = np.stack(ADE_max_scene_batch)
            ADE_max = np.hstack([ADE_max, ADE_max_scene_batch.sum(axis=0) / (sample_z)])
            ADE_avg_scene_tmp_batch = np.stack(ADE_avg_scene_tmp_batch)
            ADE_avg_scene = np.hstack([ADE_avg_scene, ADE_avg_scene_tmp_batch.sum(axis=0).sum(axis=0) / (sample*sample_z)])

            MPE_vel_min_scene_batch = np.stack(MPE_vel_min_scene_batch)
            MPE_vel_min = np.hstack([MPE_vel_min, MPE_vel_min_scene_batch.sum(axis=0) / (sample_z)])
            MPE_vel_max_scene_batch = np.stack(MPE_vel_max_scene_batch)
            MPE_vel_max = np.hstack([MPE_vel_max, MPE_vel_max_scene_batch.sum(axis=0) / (sample_z)])

            MPE_avg_scene_tmp_batch = np.stack(MPE_avg_scene_tmp_batch)
            MPE_avg_scene = np.hstack([MPE_avg_scene, MPE_avg_scene_tmp_batch.sum(axis=0).sum(axis=0) / (sample*sample_z)])
            MRE_batch = np.stack(MRE_batch)
            MRE = np.hstack([MRE, MRE_batch.sum(axis=0).sum(axis=0) / (sample*sample_z)])

            sum_timestep_KL_batch = torch.stack(sum_timestep_KL_batch)
            estimated_per_person_MutualInfo = sum_timestep_KL_batch.sum(dim=0).sum(dim=0) / (sample*sample_z)
            mutual_info = np.hstack([mutual_info, estimated_per_person_MutualInfo.cpu().numpy()])

        # test = np.vstack([mutual_info, ADE_min, ADE_max,ADE_avg_scene, MPE_vel_min, MPE_vel_max, MPE_avg_scene]).T
        # import seaborn as sns

        # df = pd.DataFrame(test, columns=['Mutual_info','ADE_min', 'ADE_max','ADE_avg_scene','MPE_vel_min','MPE_vel_max','MPE_avg_scene' ])
        # df.to_csv(config.DIR + 'SI_Policy_test_SI_50nlllcas_1'+ '/out.csv')
        # boxplot = df.boxplot(column=['Mutual_info'])
        # plot.hist(bins=12, alpha=0.5)
        # ax = df.plot.hist(bins=120, alpha=0.5)
        # ax = sns.boxplot(data=test, showfliers = False)
        # ax = sns.swarmplot(x=test)
        # plt.show()
        ADE_min_mean = np.mean(ADE_min)
        ADE_max_mean = np.mean(ADE_max)
        MPE_vel_min_mean = np.mean(MPE_vel_min)
        MPE_vel_max_mean = np.mean(MPE_vel_max)
        MPE_avg_scene = np.mean(MPE_avg_scene)
        ADE_avg_scene = np.mean(ADE_avg_scene)
        MRE = np.mean(MRE)
        FDE_mean = np.mean(FDE)
        mutual_info = np.array(mutual_info)
        act = coll_all / count_fake_all
        sum_timestep_KL_list = np.array(sum_timestep_KL_list)
        # plt.hist(mutual_info,bins='auto', density=False, range=(0,4))
        # plt.title("histogram mutual_info")
        # plt.show()
        mean_mutual_info = np.mean(mutual_info)
        # plt.hist(sum_timestep_KL_list, bins='auto', density=False, range=(0, 4))
        # plt.title("histogram timestep KL")
        # plt.show()
        return ADE_min_mean, ADE_max_mean, ADE_avg_scene, MPE_vel_min_mean,\
               MPE_vel_max_mean, MPE_avg_scene, mean_mutual_info, FDE_mean, act, MRE

def main(args, exp_name):
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
        student, robot = get_generator(checkpoint, exp_name)
        # robot = None
        _args = AttrDict(checkpoint['args'])
        # path = get_dset_path(_args.dataset_name, args.dset_type)
        path = get_dset_path('synth', args.dset_type)
        _, loader = data_loader(_args, path)
        ADE_min_mean, ADE_max_mean, ADE_avg_scene, MPE_vel_min_mean,\
        MPE_vel_max_mean, MPE_avg_scene, Mean_mutual_info,\
        FDE_mean, act, MRE = evaluate_social(_args, loader, student, robot, plot_traj=False, sample_z=5, same_goal=True)
        print(
            "Dataset:  {} \n"
            "Pred Len: {}, ADE_min_mean, ADE_max_mean, ADE_avg_scene, MPE_vel_min_mean,"
            " MPE_vel_max_mean, MPE_vel_avg_scene, Mean_mutual_info, FDE_mean, ACT, MRE \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            "{:.4f} \n"
            .format(
                _args.dataset_name, _args.pred_len, ADE_min_mean, ADE_max_mean, ADE_avg_scene,
                MPE_vel_min_mean, MPE_vel_max_mean, MPE_avg_scene, Mean_mutual_info, FDE_mean, act, MRE
            )
        )
# Here we define the prediction model for the experiments!
exp_name = 'SI-TrajNet_NoKL'
DIR = config.student_checkpoint_start_from \
                                 + 'lcas' + '/checkpoint_with_model.pt'
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=DIR, type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    args = parser.parse_args()
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    main(args, exp_name)
