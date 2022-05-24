
import logging
import sys
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import wandb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from data.loader import data_loader
from utils.losses import KL_gaussians
from utils.utils import get_dset_path,relative_to_abs
from config import *
from models.Sim2Goal import TrajectoryGenerator
from models.GoalFLow import GoalGenerator
from models.Robot import Robot

from models.Sampler import Sampler
from eval_SI import evaluate_social

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

import warnings
warnings.filterwarnings("ignore")

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)



def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


class Preparation:
    def __init__(self, data=None):
        self.config = Config()
        if data:
            self.config.dataset_name = data
            group_name = self.config.experiment_name
            self.config.experiment_name =  self.config.experiment_name + '-' + data
            path = self.config.DIR + self.config.experiment_name
            if not os.path.exists(path):
                os.makedirs(path)
            self.config.model_path = path
            self.config.output_dir = self.config.model_path
            self.config.checkpoint_start_from = self.config.model_path + '/checkpoint_with_model.pt'
            wandb.init(project="NF-Traj", name=self.config.experiment_name, reinit=True, group=group_name)
        else:
            wandb.init(project="NF-Traj", name=self.config.experiment_name, reinit=True)
            print('no_wandb')
        seed = self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu_num
        train_path = get_dset_path(self.config.dataset_name, 'train')
        val_path = get_dset_path(self.config.dataset_name, 'val')
        # test_path = get_dset_path(self.config.dataset_name, 'test')
        long_dtype, float_dtype = get_dtypes(self.config)
        logger.info("Initializing train dataset")
        self.train_dset, self.train_loader = data_loader(self.config, train_path, augment=self.config.augment)
        logger.info("Initializing val dataset")
        _, self.val_loader = data_loader(self.config, val_path)

        self.iterations_per_epoch = len(self.train_loader)
        logger.info(
            'There are {} iterations per epoch'.format(self.iterations_per_epoch)
        )

        self.robot = Robot()
        self.robot.type(float_dtype).train()
        logger.info('Here is the generator:')
        logger.info(self.robot)
        dataset_name = 'lcas'
        checkpoint_student_path = self.config.student_checkpoint_start_from \
                                 + dataset_name + '/checkpoint_with_model.pt'
        # checkpoint_student_path = self.config.student_checkpoint_start_from + '/checkpoint_with_model.pt'
        student_sampler = torch.load(checkpoint_student_path)
        self.Student = TrajectoryGenerator(self.config)
        self.Student.load_state_dict(student_sampler["best_state"])
        self.Student.cuda().eval()


        checkpoint_sampler_path = self.config.sampler_checkpoint_start_from \
                                  + dataset_name + '/checkpoint_with_model.pt'
        checkpoint_sampler = torch.load(checkpoint_sampler_path)
        self.sample_generator = GoalGenerator(self.config)
        self.sample_generator.load_state_dict(checkpoint_sampler["best_state"])
        self.sample_generator.cuda().eval()

        self.sampler = Sampler(self.config)
        self.sampler.load_state_dict(checkpoint_sampler["best_state_sampler"])
        self.sampler.cuda().eval()
        logger.info('Sampler loaded from:' + str(checkpoint_sampler['args']))



        # Log model
        # wandb.watch(self.generator, log='all')


        if self.config.adam:
            print('Learning with ADAM!')
            # betas_d = (0.5, 0.9999)
            self.optimizer = optim.Adam(self.robot.parameters(), lr=self.config.g_learning_rate)
        else:
            print('Learning with SGD!')
            self.optimizer = optim.SGD(self.robot.parameters(), lr=self.config.g_learning_rate, momentum=0.9)
        restore_path = ''
        if self.config.checkpoint_start_from:
            restore_path = self.config.checkpoint_start_from
        elif self.config.restore_from_checkpoint == True:
            restore_path = os.path.join(self.config.output_dir,
                                        '%s_with_model.pt' % self.config.checkpoint_name)

        if os.path.isfile(restore_path):
            logger.info('Restoring from checkpoint {}'.format(restore_path))
            self.checkpoint = torch.load(restore_path)
            self.robot.load_state_dict(self.checkpoint['state'])
            self.optimizer.load_state_dict(self.checkpoint['optim_state'])
            self.t = self.checkpoint['counters']['t']
            self.epoch = self.checkpoint['counters']['epoch']
            self.checkpoint['restore_ts'].append(self.t)
        else:
            # Starting from scratch, so initialize checkpoint data structure
            self.t, self.epoch = 0, 0
            self.checkpoint = {
                'args': self.config.__dict__,
                'losses': defaultdict(list),
                'losses_ts': [],
                'metrics_val': defaultdict(list),
                'metrics_train': defaultdict(list),
                'sample_ts': [],
                'restore_ts': [],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'state': None,
                'optim_state': None,
                'best_state': None,
            }

    def check_accuracy(self, loader, Student, Robot):  # TODO Change this!
        metrics = {}
        Student.eval()  # will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

        ADE_min_mean, ADE_max_mean, ADE_avg_scene, MPE_vel_min_mean, \
        MPE_vel_max_mean, MPE_vel_avg_scene, Mean_mutual_info, Mean_FDE, ACT, MRE = evaluate_social(self.config, loader, Student, Robot)
        metrics['ADE_min_mean'] = ADE_min_mean
        metrics['ADE_max_mean'] = ADE_max_mean
        metrics['ADE_avg_scene'] = ADE_avg_scene
        metrics['MPE_vel_min_mean'] = MPE_vel_min_mean
        metrics['MPE_vel_max_mean'] = MPE_vel_max_mean
        metrics['MPE_vel_avg_scene'] = MPE_vel_avg_scene
        metrics['Mean_mutual_info'] = Mean_mutual_info
        metrics['Mean_FDE'] = Mean_FDE
        metrics['ACT'] = ACT
        metrics['MRE'] = MRE
        # metrics['MPE_vel_min_mean'] = MPE_vel_min_mean
        # mean = (act+ ade)/2
        # metrics['mean'] = mean
        # Student.train()
        wandb.log({"ADE_min_mean":  metrics['ADE_min_mean'], "ADE_max_mean": metrics['ADE_max_mean'],
                   "ADE_avg_scene": metrics['ADE_avg_scene'], "MPE_vel_min_mean": metrics['MPE_vel_min_mean'],
                   "MPE_vel_max_mean": metrics['MPE_vel_max_mean'], "MPE_vel_avg_scene": metrics['MPE_vel_avg_scene'],
                   "Mean_mutual_info": metrics['Mean_mutual_info'], "Mean_FDE": metrics['Mean_FDE']
                      , "Mean_ACT": metrics['ACT'], "MRE": metrics['MRE']})
        return metrics

    def print_stats(self):
        dictlist = 'Epoch = {}, t = {} '.format(self.epoch, self.t) + '[D] '
        dictlist += ' [G] '

        for k, v in sorted(self.losses.items()):
            self.checkpoint['losses'][k].append(v)
            dictlist += ' ' + '{}: {:.6f}'.format(k, v)

        logger.info(dictlist)

        self.checkpoint['losses_ts'].append(self.t)

    def save_model(self):
        self.checkpoint['counters']['t'] = self.t
        self.checkpoint['counters']['epoch'] = self.epoch
        self.checkpoint['sample_ts'].append(self.t)

        # Check stats on the validation set
        logger.info('Checking stats on val ...')
        metrics_val = self.check_accuracy(self.val_loader, self.Student, self.robot)
        # metrics_va_gt = self.check_accuracy_gt(self.val_loader, self.Student, self.robot)
        # metrics_val.update(metrics_va_gt)
    #    self.scheduler.step(metrics_val['ade'])
        for k, v in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(k, v))
            self.checkpoint['metrics_val'][k].append(v)

        min_mean = min(self.checkpoint['metrics_val']['Mean_mutual_info'])
        if metrics_val['Mean_mutual_info'] == min_mean:
            logger.info('New low for mean error')
            self.checkpoint['best_t'] = self.t
            self.checkpoint['best_state'] = copy.deepcopy(self.robot.state_dict())

        # Save another checkpoint with model weights and
        # optimizer state
        self.checkpoint['state'] = self.robot.state_dict()
        self.checkpoint['optim_state'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(
            self.config.output_dir, '%s_with_model.pt' % self.config.checkpoint_name
        )
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.checkpoint, checkpoint_path)
        torch.save(self.checkpoint, os.path.join(wandb.run.dir, 'model.pt'))
        logger.info('Done.')

        # Save a checkpoint with no model weights by making a shallow
        # copy of the checkpoint excluding some items
        checkpoint_path = os.path.join(
            self.config.output_dir, '%s_no_model.pt' % self.config.checkpoint_name)
        logger.info('Saving checkpoint to {}'.format(checkpoint_path))
        key_blacklist = [
            'state', 'best_state', 'optim_state'
        ]
        small_checkpoint = {}
        for k, v in self.checkpoint.items():
            if k not in key_blacklist:
                small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)
        logger.info('Done.')
    # for debugging
    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()

    def model_step(self, batch):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask,\
        loss_mask, seq_start_end, nei_num_index, nei_num, = batch
        robots_ids = seq_start_end[:,0]
        losses = {}
        MSE_loss = nn.MSELoss()
        for param in self.robot.parameters(): param.grad = None
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

        sgoals = pred_traj_gt[-1]
        # current_pos = obs_traj[-1]
        # dist_mat_pred = torch.cdist(sgoals, current_pos, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
        # for _, (start, end) in enumerate(seq_start_end):
        #     start = start.item()
        #     end = end.item()
        #     dist_mat_pred_scene = dist_mat_pred[start:end, start]
        #     dist_mat_pred_scene[0] = np.inf
        #     min_index = torch.argmin(dist_mat_pred_scene)
        #     sgoals[start] = sgoals[min_index + start]
        # sgoals[robots_ids] = sgoals[seq_start_end[:, 1] - 1]
        z = torch.randn_like(pred_traj_gt)
        pred_traj_fake_rel_c, mu_c, logscale_c, nll_robot,  pred_traj_fake_rel_sampled_c = self.Student(model_input, obs_traj, pred_traj_gt,
                                                                 seq_start_end, nei_num_index,nei_num, mode='robot_train',
                                                                 sample_goal=sgoals,robot_net=self.robot,
                                                                 robotID=robots_ids, noise=z, detached=False)
        pred_traj_fake_abs = relative_to_abs(pred_traj_fake_rel_c, obs_traj[-1])
        pred_traj_fake_abs_sampled = relative_to_abs(pred_traj_fake_rel_sampled_c, obs_traj[-1])
        mu_c_offset = torch.cat([obs_traj[-1].unsqueeze(dim=0), pred_traj_fake_abs], dim=0)
        mu_c = mu_c + mu_c_offset[0:-1]

        end_pos = pred_traj_fake_abs_sampled[-1]
        goal_loss = MSE_loss(end_pos,sgoals)
        ids_no_robot = torch.ones([seq_start_end[-1, 1]], dtype=torch.bool).cuda()
        ids_no_robot[robots_ids] = False

        nei_num_index_tmp = nei_num_index[:, ids_no_robot]
        nei_num_index_tmp = nei_num_index_tmp[:, :, ids_no_robot]
        model_input_tmp = model_input[:, ids_no_robot]
        obs_traj_tmp = obs_traj[:, ids_no_robot]
        pred_traj_gt_tmp = pred_traj_gt[:, ids_no_robot]
        sgoals_tmp = sgoals[ids_no_robot]
        unconditioned_vel, mu_unc, logscale_unc = self.Student(model_input_tmp, obs_traj_tmp, pred_traj_gt_tmp,
                                                          seq_start_end, nei_num_index_tmp, nei_num,
                                                          mode='test', sample_goal=sgoals_tmp,
                                                          noise=z[:, ids_no_robot], detached=False)

        unconditioned_pos = relative_to_abs(unconditioned_vel, obs_traj_tmp[-1])
        mu_offset = torch.cat([obs_traj_tmp[-1].unsqueeze(dim=0), unconditioned_pos], dim=0)
        mu_unc = mu_unc + mu_offset[0:-1]

        sum_timestep_KL = KL_gaussians(mu_c[:, ids_no_robot], logscale_c[ :, ids_no_robot],
                                       mu_unc, logscale_unc, sum_dim=0).sum()

        loss = self.config.g_w * goal_loss + self.config.si_w * sum_timestep_KL + self.config.nll_w * nll_robot

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.robot.parameters(), max_norm=3.0, norm_type=2)
        self.optimizer.step()
        return losses

    def train(self):
        self.t_step = 0
        while self.epoch < self.config.num_epochs:
            self.t_step = 0
            logger.info('Starting epoch {}'.format(self.epoch))
            for j, batch in enumerate(self.train_loader):
                batch = [tensor.cuda() for tensor in batch]
                self.losses = self.model_step(batch)
                self.t_step += 1
            if self.epoch % self.config.check_after_num_epochs == 0:
                self.save_model()
            self.epoch += 1

if __name__ == '__main__':
    prep = Preparation()
    prep.train()