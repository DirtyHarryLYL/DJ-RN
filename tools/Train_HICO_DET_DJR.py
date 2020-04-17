
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import cPickle as pickle
import os

from ult.config import cfg
from models.train_Solver_HICO_DET_DJR import train_net
from networks.DJR import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=1800000, type=int)
    parser.add_argument('--iter', dest='iter', help='iter to load', default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='DJR', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one. (By jittering the object detections)',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)
    parser.add_argument('--train_continue', dest='train_continue',
            help='Whether to continue from previous ckpt',
            default=cfg.TRAIN_MODULE_CONTINUE, type=int)
    parser.add_argument('--init_weight', dest='init_weight',
            help='How to init weight',
            default=3, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg.TRAIN_MODULE_CONTINUE   = args.train_continue
    cfg.TRAIN_INIT_WEIGHT       = args.init_weight

    Trainval_GT       = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_idx.pkl', "rb"))
    Trainval_N        = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_idx.pkl', "rb")) 

    np.random.seed(cfg.RNG_SEED)
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        weight    = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_%d.ckpt' % args.iter
    else:
        if cfg.TRAIN_INIT_WEIGHT == 3:
            weight    = cfg.ROOT_DIR + '/Weights/TIN_HICO/HOI_iter_1700000.ckpt' 
        else:
            weight = None
    
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

    net = ResNet50()
 
    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, args.Restore_flag, weight, max_iters=args.max_iters)
