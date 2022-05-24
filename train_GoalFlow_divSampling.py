
import logging
import sys
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim
import copy
import wandb
from data.loader import data_loader
from utils.utils import get_dset_path
from config import *
from models.GoalFLow import GoalGenerator
from models.Sampler import Sampler
from eval_GoalFlow_divSampling import evaluate

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
        checkpoint_glow_path = self.config.GFlow_checkpoint_start_from \
                                  + self.config.dataset_name + '/checkpoint_with_model.pt'
        checkpoint_glow = torch.load(checkpoint_glow_path)
        self.generator = GoalGenerator(self.config)
        self.generator.load_state_dict(checkpoint_glow["best_state"])
        self.generator.type(float_dtype).train()
        logger.info('Here is the Goal Network loaded from: ' + checkpoint_glow_path)
        logger.info(self.generator)

        self.sampler = Sampler(self.config)
        self.sampler.type(float_dtype).train()
        logger.info('Here is the Sampler:')
        logger.info(self.sampler)


        # Log model
        # wandb.watch(self.generator, log='all')

        restore_path = None

        if self.config.restore_from_checkpoint == True:
            restore_path = os.path.join(self.config.output_dir,
                                        '%s_with_model.pt' % self.config.checkpoint_name)
            if os.path.isfile(restore_path):
                logger.info('Restoring from checkpoint {}'.format(restore_path))
                self.checkpoint = torch.load(restore_path)
                self.sampler.load_state_dict(self.checkpoint['best_state'])
                # self.optimizer.load_state_dict(self.checkpoint['optim_state'])
                # self.t = self.checkpoint['counters']['t']
                # self.t, self.epoch = 0, 0
                # self.epoch = self.checkpoint['counters']['epoch']
                # self.checkpoint['restore_ts'].append(self.t)
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
        if self.config.adam:
            print('Learning with SGD!')
            # betas_d = (0.5, 0.9999)
            self.optimizer = optim.SGD(self.sampler.parameters(), lr=self.config.g_learning_rate)
            # self.optimizer = optim.Adam(self.sampler.parameters(), lr=self.config.g_learning_rate)
        else:
            print('Learning with RMSprop!')
            self.optimizer = optim.SGD(self.sampler.parameters(), lr=0.001, momentum=0.9)

    def check_accuracy(self, loader, generator, sampler):  # TODO Change this!
        metrics = {}
        generator.eval()  # will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
        sampler.eval()
        fde = evaluate(self.config, loader, generator, sampler)
        metrics['fde'] = fde
        generator.train()
        wandb.log({"fde": metrics['fde']})
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
        metrics_val = self.check_accuracy(self.val_loader, self.generator, self.sampler)
    #    self.scheduler.step(metrics_val['ade'])
        for k, v in sorted(metrics_val.items()):
            logger.info('  [val] {}: {:.3f}'.format(k, v))
            self.checkpoint['metrics_val'][k].append(v)

        min_fde = min(self.checkpoint['metrics_val']['fde'])
        if metrics_val['fde'] == min_fde:
            logger.info('New low for fde error')
            self.checkpoint['best_t'] = self.t
            self.checkpoint['best_state'] = copy.deepcopy(self.generator.state_dict())
            self.checkpoint['best_state_sampler'] = copy.deepcopy(self.sampler.state_dict())

        # Save another checkpoint with model weights and
        # optimizer state
        self.checkpoint['state'] = self.generator.state_dict()
        self.checkpoint['optim_state'] = self.optimizer.state_dict()
        self.checkpoint['state_sampler'] = self.sampler.state_dict()
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
        l2_loss_rel = []
        losses = {}
        for param in self.generator.parameters(): param.grad = None

        # loss_mask = loss_mask[-self.config.pred_len :]
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)


        pred_traj_fake_rel, div_loss = self.generator(model_input, obs_traj, pred_traj_gt,
                                                      seq_start_end, nei_num_index, nei_num,
                                                      mode='sampling', sampling_module=self.sampler)

        loss = div_loss
        loss.backward()
        losses['div_loss'] = div_loss.item()
        if self.config.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()
        wandb.log({"div_loss": div_loss.item()})
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
            # if self.epoch == 40:
            #     self.config.check_after_num_epochs = 1

if __name__ == '__main__':
    prep = Preparation()
    prep.train()