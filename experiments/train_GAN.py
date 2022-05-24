
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
from utils.losses import l2_loss, coll_smoothed_loss, compute_gradient_penalty1D
from utils.utils import get_dset_path
from config import *
from models.CoLoss_GAN import TrajectoryGenerator, Discriminator

from evaluate_GAN import evaluate
from utils.utils import relative_to_abs
from torch.autograd import Variable
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

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
        self.discriminator = Discriminator(self.config)
        self.discriminator.type(float_dtype).train()
        logger.info('Here is the discriminator:')
        logger.info(self.discriminator)
        # Log model
        # wandb.watch(self.generator, log='all')


        if self.config.adam:
            print('Learning with ADAM!')
            betas_d = (0.5, 0.9999)
            self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.config.g_learning_rate, betas=betas_d)
            self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.config.d_learning_rate, betas=betas_d)
        else:
            print('Learning with RMSprop!')
            self.optimizer_g = optim.SGD(self.generator.parameters(), lr=0.01, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, 'min')
        restore_path = None
        if self.config.checkpoint_start_from is not None:
            restore_path = self.config.checkpoint_start_from
        elif self.config.restore_from_checkpoint == True:
            restore_path = os.path.join(self.config.output_dir,
                                        '%s_with_model.pt' % self.config.checkpoint_name)

        if os.path.isfile(restore_path):
            logger.info('Restoring from checkpoint {}'.format(restore_path))
            self.checkpoint = torch.load(restore_path)
            self.generator.load_state_dict(self.checkpoint['g_state'])
            self.discriminator.load_state_dict(self.checkpoint['d_state'])
            self.optimizer_g.load_state_dict(self.checkpoint['g_optim_state'])
            self.optimizer_d.load_state_dict(self.checkpoint['d_optim_state'])
            self.t = self.checkpoint['counters']['t']
            self.epoch = self.checkpoint['counters']['epoch']
            self.checkpoint['restore_ts'].append(self.t)
        else:
            # Starting from scratch, so initialize checkpoint data structure
            self.t, self.epoch = 0, 0
            self.checkpoint = {
                'args': self.config.__dict__,
                'G_losses': defaultdict(list),
                'D_losses': defaultdict(list),
                'losses_ts': [],
                'metrics_val': defaultdict(list),
                'metrics_train': defaultdict(list),
                'sample_ts': [],
                'restore_ts': [],
                'counters': {
                    't': None,
                    'epoch': None,
                },
                'g_state': None,
                'g_optim_state': None,
                'g_best_state': None,
                'd_state': None,
                'd_optim_state': None,
                'd_best_state': None,
            }

    def check_accuracy(self, loader, generator):  # TODO Change this!
        metrics = {}
        generator.eval()  # will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

        ade, fde, act, _ = evaluate(self.config, loader, generator)
        metrics['act'] = act
        metrics['ade'] = ade
        metrics['fde'] = fde
        mean = (act+ ade)/2
        metrics['mean'] = mean
        generator.train()
        wandb.log({"ade":  metrics['ade'], "fde": metrics['fde'],
                   "act_best_ade": metrics['act'], "Mean_ade_act": metrics['mean']})
        return metrics

    def print_stats(self):
        dictlist = 'Epoch = {}, t = {} '.format(self.epoch, self.t) + '[D] '
        dictlist += ' [G] '

        for k, v in sorted(self.losses_g.items()):
            self.checkpoint['G_losses'][k].append(v)
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
            self.checkpoint['g_best_state'] = copy.deepcopy(self.generator.state_dict())
            self.checkpoint['d_best_state'] = self.discriminator.state_dict()

        # Save another checkpoint with model weights and
        # optimizer state
        self.checkpoint['g_state'] = self.generator.state_dict()
        self.checkpoint['g_optim_state'] = self.optimizer_g.state_dict()
        self.checkpoint['d_state'] = self.discriminator.state_dict()
        self.checkpoint['d_optim_state'] = self.optimizer_g.state_dict()
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
            'g_state', 'g_best_state', 'g_optim_state',
            'd_state', 'd_best_state', 'd_optim_state'
        ]
        small_checkpoint = {}
        for k, v in self.checkpoint.items():
            if k not in key_blacklist:
                small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)
        logger.info('Done.')

    def discriminator_step(self, batch):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask, \
        loss_mask, seq_start_end, nei_num_index, nei_num = batch
        losses = {}
        # self.discriminator.zero_grad()
        loss_mask = loss_mask[-self.config.pred_len:]
        for param in self.discriminator.parameters(): param.grad = None # more eff than self.discriminator.zero_grad()
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
        pred_traj_fake_rel = self.generator(model_input, obs_traj, pred_traj_gt,
                                            seq_start_end, nei_num_index, nei_num, 0).detach()
        scores_real = self.discriminator(obs_traj_rel, pred_traj_gt_rel,
                                         obs_traj,nei_num_index, nei_num, loss_mask)
        scores_fake = self.discriminator(obs_traj_rel, pred_traj_fake_rel,
                                         obs_traj,nei_num_index, nei_num, loss_mask)
        if self.config.loss == 'hinge':
            disc_loss = nn.ReLU()(1.0 - scores_real).mean() + nn.ReLU()(1.0 + scores_fake).mean()
        elif self.config.loss == 'wasserstein':
            disc_loss = -scores_real.mean() + scores_fake.mean()
        else:
            disc_loss = nn.BCEWithLogitsLoss()(scores_real, Variable(torch.ones(scores_real.size()[0], 1).cuda())) + \
                        nn.BCEWithLogitsLoss()(scores_fake, Variable(torch.zeros(scores_real.size()[0], 1).cuda()))
        loss = disc_loss
        if self.config.cpg_loss:
            cgp_loss = self.config.cpg_loss * compute_gradient_penalty1D(self.discriminator, obs_traj_rel, pred_traj_gt_rel, pred_traj_fake_rel,
                                                                         obs_traj, nei_num_index, nei_num, loss_mask)
            loss = loss + cgp_loss
            # cgp_loss.backward()
        loss.backward()
        losses['D_loss'] = disc_loss.item()
        self.optimizer_d.step()
        wandb.log({"D_loss": disc_loss.item()})
        return losses

    def gen_step(self, batch):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, val_mask,\
        loss_mask, seq_start_end, nei_num_index, nei_num, = batch
        l2_loss_rel = []
        losses = {}
        for param in self.generator.parameters(): param.grad = None

        loss_mask = loss_mask[-self.config.pred_len :]
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
        for _ in range(self.config.best_k):
            pred_traj_fake_rel = self.generator(model_input, obs_traj, pred_traj_gt,
                                                seq_start_end, nei_num_index, nei_num, 0)
            l2_loss_rel.append(
                l2_loss(
                    pred_traj_fake_rel,
                    model_input[-self.config.pred_len :],
                    loss_mask,
                    mode="raw",
                )
            )
        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
            _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
            _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_traj_fake_rel.shape[0]) * (end - start)
            )
            l2_loss_sum_rel += _l2_loss_rel

        if self.config.loss == 'hinge' or self.config.loss == 'wasserstein':
            scores_fake = -self.discriminator(obs_traj_rel, pred_traj_fake_rel,
                                              obs_traj, nei_num_index, nei_num, loss_mask).mean() #+ loss_l2*100
        else:
            scores_fake = nn.BCEWithLogitsLoss()(self.discriminator(obs_traj_rel, pred_traj_fake_rel,
                                                                    obs_traj,nei_num_index, nei_num, loss_mask),
                                                 Variable(torch.ones(obs_traj.size()[1], 1).cuda()))

        pred_traj_fake_abs = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        loss_count = coll_smoothed_loss(pred_traj_fake_abs, seq_start_end, nei_num_index)
        loss = l2_loss_sum_rel + scores_fake + 10 * loss_count
        # loss = scores_fake
        loss.backward()
        losses['L2_loss'] = l2_loss_sum_rel.item()
        self.optimizer_g.step()
        wandb.log({"Train L2": l2_loss_sum_rel.item()})
        return losses

    def train(self):
        self.t_step = 0
        while self.epoch < self.config.num_epochs:
            d_steps_left = self.config.d_steps
            g_steps_left = self.config.g_steps
            self.t_step = 0
            logger.info('Starting epoch {}'.format(self.epoch))
            for batch in self.train_loader:
                batch = [tensor.cuda() for tensor in batch]
                if g_steps_left > 0:
                    self.losses_g = self.gen_step(batch)
                    g_steps_left -= 1
                elif d_steps_left > 0:
                    self.losses_d = self.discriminator_step(batch)
                    d_steps_left -= 1
                self.t += 1
                if d_steps_left > 0 or g_steps_left > 0:
                    continue
                d_steps_left = self.config.d_steps
                g_steps_left = self.config.g_steps
                self.t_step += 1
            if self.epoch % self.config.check_after_num_epochs == 0:
                self.save_model()
            self.epoch += 1


if __name__ == '__main__':
    prep = Preparation()
    prep.train()