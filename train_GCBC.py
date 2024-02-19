
import logging
import sys
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import wandb
from data.loader import data_loader
from utils.losses import coll_smoothed_loss
from utils.utils import get_dset_path
from config import *

from models.Goal_Transformer import TrajectoryGenerator

from eval_Sim2Goal import evaluate
from utils.utils import relative_to_abs

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
            # print('no_wandb')
        seed = self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu_num
        train_path = get_dset_path(self.config.dataset_name, 'train')
        val_path = get_dset_path(self.config.dataset_name, 'test')
        long_dtype, float_dtype = get_dtypes(self.config)
        logger.info("Initializing train dataset")
        self.train_dset, self.train_loader = data_loader(self.config, train_path, augment=self.config.augment)
        logger.info("Initializing val dataset")
        _, self.val_loader = data_loader(self.config, val_path)
        self.iterations_per_epoch = len(self.train_loader)
        logger.info(
            'There are {} iterations per epoch'.format(self.iterations_per_epoch)
        )

        self.generator = TrajectoryGenerator(self.config)
        self.generator.type(float_dtype).train()
        logger.info('Here is the generator:')
        logger.info(self.generator)

        if self.config.adam:
            print('Learning with ADAM!')
            # betas_d = (0.5, 0.9999)
            self.optimizer = optim.Adam(self.generator.parameters(), lr=self.config.g_learning_rate)
        else:
            print('Learning with SGD!')
            self.optimizer = optim.SGD(self.generator.parameters(), lr=self.config.g_learning_rate, momentum=0.9)
        restore_path = None
        if self.config.checkpoint_start_from is not None:
            restore_path = self.config.checkpoint_start_from
        elif self.config.restore_from_checkpoint == True:
            restore_path = os.path.join(self.config.output_dir,
                                        '%s_with_model.pt' % self.config.checkpoint_name)

        if os.path.isfile(restore_path):
            logger.info('Restoring from checkpoint {}'.format(restore_path))
            self.checkpoint = torch.load(restore_path)
            self.generator.load_state_dict(self.checkpoint['state'])
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

    def check_accuracy(self, loader, generator): 
        metrics = {}
        generator.eval()  # will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

        ade, fde, act, _ = evaluate(self.config, loader, generator)
        metrics['act'] = act
        metrics['ade'] = ade
        metrics['fde'] = fde
        mean = (act + fde)/2
        metrics['mean'] = mean
        generator.train()
        wandb.log({"ade":  metrics['ade'], "fde": metrics['fde'],
                   "act_best_ade": metrics['act'], "Mean_ade_act": metrics['mean']})
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
        metrics_val = self.check_accuracy(self.val_loader, self.generator)
    #    self.scheduler.step(metrics_val['ade'])
        for k, v in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(k, v))
            self.checkpoint['metrics_val'][k].append(v)

        min_mean = min(self.checkpoint['metrics_val']['mean'])
        if metrics_val['mean'] == min_mean:
            logger.info('New low for mean error')
            self.checkpoint['best_t'] = self.t
            self.checkpoint['best_state'] = copy.deepcopy(self.generator.state_dict())

        # Save another checkpoint with model weights and
        # optimizer state
        self.checkpoint['state'] = self.generator.state_dict()
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

    def model_step(self, batch):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask,\
        loss_mask, seq_start_end, nei_num_index, nei_num, = batch

        losses = {}
        MSE_loss = nn.MSELoss()
        for param in self.generator.parameters(): param.grad = None

        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

        # goals_ids = np.expand_dims(np.random.choice(self.config.pred_len, pred_traj_gt.shape[1]), axis=1)
        goals_ids = torch.tensor(np.random.choice(self.config.pred_len, pred_traj_gt.shape[1]))
        goal = pred_traj_gt[goals_ids, torch.arange(pred_traj_gt.shape[1]), :]
        pred_traj_fake_rel, nll, scale_sum = self.generator(model_input, obs_traj, pred_traj_gt,
                                                                 seq_start_end, nei_num_index,
                                                 nei_num, mode='train',sample_goal=goal)
        #pred_traj_fake_abs = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        #loss_count = coll_smoothed_loss(pred_traj_fake_abs, seq_start_end, nei_num_index)

        loss = nll #+ 30. * loss_count + 0.01*scale_sum# + corrected_pos_loss
        loss.backward()
        losses['nll'] = nll.item()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=3.0, norm_type=2)
        self.optimizer.step()
        wandb.log({"nll": nll.item()})
        return losses

    def train(self):
        self.t_step = 0
        while self.epoch < self.config.num_epochs:
            self.t_step = 0
            logger.info('Starting epoch {}'.format(self.epoch))
            for batch in self.train_loader:
                batch = [tensor.cuda() for tensor in batch]
                self.losses = self.model_step(batch)
                self.t_step += 1
            if self.epoch % self.config.check_after_num_epochs == 0:
                self.save_model()
            self.epoch += 1

if __name__ == '__main__':
    prep = Preparation()
    prep.train()