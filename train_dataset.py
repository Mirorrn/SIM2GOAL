# Choose what to train, do not forget to set the right config!
# from train_GoalFlow import Preparation
# from train_GoalFlow_divSampling import Preparation
from train_Sim2Goal import Preparation
# from experiments.train_GAN import Preparation
# from experiments.train_VAE import Preparation
# from experiments.train_SimNoGoal import Preparation

from config import Config
config = Config()

if config.trajnet:
    datasets = ['wildtrack', 'students1', 'students3', 'zara1', 'zara3', 'hotel', 'lcas']
else:
    datasets = ['hotel', 'zara1', 'zara2', 'univ', 'eth']  # eth & utc
if __name__ == '__main__':
    for data in datasets:
        prep = Preparation(data)
        prep.train()
        del prep
        print('START NEW DATASET ####################################################')