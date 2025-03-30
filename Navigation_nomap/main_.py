import os
from trainer import PPOTrainer
from dataloader import StreetLearnDataset
import argparse
import warnings
from utils import Create_landmark
warnings.filterwarnings('ignore')

from agent import *
import numpy as np

parser = argparse.ArgumentParser( description='Navigation')

parser.add_argument('--data_root', default='./data', type=str)
parser.add_argument('--save_dir', default='./results')
parser.add_argument('--train_model', default='ConvLSTM')
parser.add_argument('--load_model', default='best', type=str, help="load pretrained model for test or training")

parser.add_argument('--height', default=84, type=int, help='time length')
parser.add_argument('--width', default=84, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--device', default='cuda:1', type=str)

parser.add_argument('--feats_in', default=3, type=int)
parser.add_argument('--feats_h', default=64, type=int)
parser.add_argument('--feats_out', default=4, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--test_rate', default=0.2, type=float)

parser.add_argument('--gamma', default=0.99, type=float, help='reward for future')
parser.add_argument('--gae_lambda', default=0.95, type=float, help='TD error and Monte Carlo, 0-1, 1 is Monte Carlo')
parser.add_argument('--clip_epsilon', default=0.2, type=float, help='strategic shear, 0.1-0.3. 0.3 exploratory and 0.1 stable')
parser.add_argument('--ent_coef', default=0.01, type=float,  help='entropy coefficient, 0.0001-0.1. 0.001 strategy is more deterministic, 0.1 more stochastic')

parser.add_argument('--num_landmarks', default=50, type=int)
parser.add_argument('--max_steps', default=200, type=int)
parser.add_argument('--num_episodes', default=10, type=int)
parser.add_argument('--total_timesteps', default=100000, type=int)
parser.add_argument('--vision_encoder', default=False, type=bool)
parser.add_argument('--kernel_size', default=8, type=int)
parser.add_argument('--stride', default=4, type=int)
parser.add_argument('--emb_coe', default=0.002, type=float)




args = parser.parse_args()



if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.data_root + '/landmark.npy'):
    Create_landmark()


print('Loading data #################################')
dataset = StreetLearnDataset(root_dir= args.data_root +'/streetlearn_data')

ppo = PPOTrainer(args, dataset)

print('Train start #################################')
ppo.train(total_timesteps=args.total_timesteps)





