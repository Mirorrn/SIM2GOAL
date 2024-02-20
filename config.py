
import os
from os.path import expanduser
home = expanduser("~")


class Config:
    DIR = home +'/SIM2GOAL/models/weights/'
    experiment_name = 'GCBC-univ_AR_Transformer_more_data_debug'
    path= DIR + experiment_name
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = path
    # Dataset options
    dataset_name = 'univ'   # Choose dataset split
    goal_sampling = False
    trajnet = False # Trajenet++ training datasetset will be used
    nabs = False # absolute position but shift the origin to the latest observed time slot
    delim = 'tab'
    loader_num_workers = 4
    # ETH & UTC
    if trajnet:
        obs_len = 9
        pred_len = 12
    else:
        obs_len = 8
        pred_len = 12
    # only for ETH and UTC
    skip = 1
    seed = 42
    min_x = -7.69
    min_y = -10.31
    img_h = 28
    img_w = 28
    collision_distance = 0.2
    # Model Options
    prelearning = False

    cpg_loss = 10.
    l2_loss = 10.
    best_k = 1.
    num_samples = 20

    # for Goal Flow Training
    clamp_flow = False
    clip_grad = False

    # Generator Options
    g_learning_rate = 0.001    # For Coloss-GAN and all other generators learning rate
    g_steps = 1                # generator step
    # Discriminator Options
    adam = 1                  # check in code (Preperation) what else will be used, div sampler sgd will be used
    augment = True
    all_rel_persons = False
    gpu_num = "0"
    device='cpu'
    # Optimization
    batch_size = 64
    num_epochs = 300
    # Loss Options
    l2_loss_weight = 1
    # Output
    output_dir = model_path
    checkpoint_name = 'checkpoint'
    restore_from_checkpoint = False
    checkpoint_start_from = ''

    if not trajnet:
        GFlow_checkpoint_start_from   = DIR + 'GFLOW-ETHandUCY-'
        sampler_checkpoint_start_from = DIR + 'GFLOW-ETHandUCY_sampler-'

    else:
        sampler_checkpoint_start_from = DIR + 'GFLOW-TrajNet_sampler-'
        GFlow_checkpoint_start_from   = DIR + 'GFLOW-TrajNet-'
        student_checkpoint_start_from = DIR + 'SIM2Goal-TrajNet-'

    num_samples_check = 5000
    check_after_num_epochs = 5
    # Misc
    use_gpu = 1
    timing = 0
    passing_time = 1
    ifdebug = False