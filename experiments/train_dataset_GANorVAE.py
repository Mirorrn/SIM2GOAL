from train_GAN import Preparation
from config import Config
config = Config()

if config.trajnet:
    # datasets = ['students1','students3', 'zara1', 'zara3', 'hotel','lcas']
    datasets = ['wildtrack', 'students1', 'students3', 'zara1', 'zara3', 'hotel', 'lcas']
else:
    datasets = ['eth', 'zara1', 'zara2', 'univ', 'hotel']  # eth & utc
if __name__ == '__main__':
    for data in datasets:
        prep = Preparation(data)
        prep.train()
        del prep
        print('START NEW DATASET ####################################################')