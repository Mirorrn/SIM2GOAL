import math
import os
from os.path import expanduser
home = expanduser("~")

class Config:
    def __init__(self):
        self.DIR = home +'/Sim2Goal/models/weights/'
        self.experiment_name = 'SIM2Goal-univ_fast_Transformer_debug3'
        path= self.DIR + self.experiment_name
        if not os.path.exists(path):
            os.makedirs(path)
        self.model_path = path
        # Dataset options
        self.dataset_name = 'univ'   # Choose dataset split
        self.goal_sampling = False
        self.trajnet = False # Trajenet++ training datasetset will be used
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
        # only for ETH and UTC
        self.skip = 1
        self.seed = 42
        self.min_x = -7.69
        self.min_y = -10.31
        self.img_h = 28
        self.img_w = 28
        self.collision_distance = 0.2
        # Model Options
        self.prelearning = False

        self.cpg_loss = 10.
        self.l2_loss = 10.
        self.best_k = 1.
        self.num_samples = 20

        # for Goal Flow Training
        self.clamp_flow = False
        self.clip_grad = False

        # Generator Options
        self.g_learning_rate = 0.001    # For Coloss-GAN and all other generators learning rate
        self.g_steps = 1                # generator step
        # Discriminator Options
        self.adam = 1                  # check in code (Preperation) what else will be used, div sampler sgd will be used
        self.augment = True
        self.all_rel_persons = False
        self.gpu_num = "0"
        self.device='cpu'
        # Optimization
        self.batch_size = 64
        self.num_epochs = 300
        # Loss Options
        self.l2_loss_weight = 1
        # Output
        self.output_dir = self.model_path
        self.checkpoint_name = 'checkpoint'
        self.restore_from_checkpoint = False
        self.checkpoint_start_from = ''

        if not self.trajnet:
           self.GFlow_checkpoint_start_from   = self.DIR + 'GFLOW-ETHandUCY-'
           self.sampler_checkpoint_start_from = self.DIR + 'GFLOW-ETHandUCY_sampler-'

        else:
            self.sampler_checkpoint_start_from = self.DIR + 'GFLOW-TrajNet_sampler-'
            self.GFlow_checkpoint_start_from   = self.DIR + 'GFLOW-TrajNet-'
            self.student_checkpoint_start_from = self.DIR + 'SIM2Goal-TrajNet-'

        self.num_samples_check = 5000
        self.check_after_num_epochs = 5
        # Misc
        self.use_gpu = 1
        self.timing = 0
        self.passing_time = 1
        self.ifdebug = False