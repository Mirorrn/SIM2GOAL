import argparse
import torch
from config import *
config = Config()
"""
Tiny utility to print the command-line args used for a checkpoint
"""

parser = argparse.ArgumentParser()
#parser.add_argument('--checkpoint', default='/home/martin-pc/PycharmProjects/crowd-sim/models/sgan-p-models/zara2_8_model.pt')
#parser.add_argument('--model_path',default='/home/martin/sgan/models/sgan-p-models/hotel_8_model.pt', type=str)
DIR = home +'/Sim2Goal/models/weights/SIM2Goal-TrajNet-wildtrack/checkpoint_with_model.pt'
# DIR = '/home/WIN-UNI-DUE/adi205v/Documents/NFTraj/SIM2Goal-TrajNet_Clip_Clamp_more_epochs-wildtrack/checkpoint_with_model.pt'
parser.add_argument('--checkpoint',default=DIR, type=str)


def main(args):
	checkpoint = torch.load(args.checkpoint, map_location='cpu')
	for k, v in checkpoint['args'].items():
		print(k, v)


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)