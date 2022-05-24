import math
import os
from os.path import expanduser
home = expanduser("~")

class Config:
    def __init__(self):
        self.DIR = home +'/Documents/NFTraj/'
        self.experiment_name = 'VAE_Trajnet'
        path= self.DIR + self.experiment_name
        if not os.path.exists(path):
            os.makedirs(path)
        self.model_path = path
        # Dataset options
        self.dataset_name = 'wildtrack'   # synth
        self.trajnet = True # Trajenet++ training datasetset will be used
        self.nabs = False # absolute position but shift the origin to the latest observed time slot
        self.delim = 'tab'
        self.loader_num_workers = 4
        # ETH & UTC
        if self.trajnet:
            self.obs_len = 9
            self.pred_len = 12
        else:
            self.obs_len = 8
            self.pred_len = 12
        self.skip = 1
        self.seed = 42
        self.min_x = -7.69
        self.min_y = -10.31
        self.img_h = 28
        self.img_w = 28
        self.collision_distance = 0.2
        # Model Options
        self.prelearning = False
        self.initial_mmd = 10
        self.MMD_multiplier = 0
        self.max_mmd = 500
        self.alpha_epoch_start = 0
        self.alpha_multiplier = 1e-3
        self.cpg_loss = 10.
        self.l2_loss = 10.
        self.best_k = 1
        self.num_samples = 20
        # Generator Options
        self.latent_dim = 25
        self.cond_dropout = 0
        self.pl_dropout = 0.1
        self.softmax_dec = True
        self.g_learning_rate = 0.001
        self.g_steps = 1                # generator step
        self.adam = 1
        self.augment = True
        self.all_rel_persons = False   # if True consider all persons in a time window
        self.gpu_num = "0"
        self.device='cuda'
        # Optimization
        self.batch_size = 32
        self.num_epochs = 100
        # Output
        self.output_dir = self.model_path
        self.checkpoint_name = 'checkpoint'
        self.restore_from_checkpoint = False
        self.checkpoint_start_from = ''



        self.num_samples_check = 5000
        self.check_after_num_epochs = 5
        # Misc
        self.use_gpu = 1
        self.timing = 0
        self.passing_time = 1
        self.ifdebug = False