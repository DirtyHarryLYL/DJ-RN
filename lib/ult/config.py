from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.TRAIN = edict()
__C.TRAIN_MODULE = 1
__C.TRAIN_MODULE_UPDATE = 1
__C.TRAIN_INIT_WEIGHT = 3
__C.TRAIN_MODULE_CONTINUE = 2
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN_DROP_OUT_BINARY = 0.8
__C.TRAIN.SNAPSHOT_ITERS = 100000
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.GAMMA = 0.96
__C.TRAIN.STEPSIZE = 20000
__C.TRAIN.SNAPSHOT_KEPT = None
__C.TRAIN.DISPLAY = 10
__C.TRAIN.SUMMARY_INTERVAL = 200
__C.RESNET = edict()
__C.RESNET.MAX_POOL = False
__C.RESNET.FIXED_BLOCKS = 1
__C.LR_DECAY = edict()
__C.LR_DECAY.TYPE = 'none'
__C.LR_DECAY.STEPS = 5.0
__C.LR_DECAY.T_MUL = 2.0
__C.LR_DECAY.M_MUL = 1.0
__C.LANG_NOISE = 0
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.RNG_SEED = 3
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'Data'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'Data', 'smplx_res'))
__C.EXP_DIR = 'default'
__C.USE_GPU_NMS = True
__C.POOLING_MODE = 'crop'
__C.POOLING_SIZE = 7
__C.ANCHOR_SCALES = [8,16,32]
__C.ANCHOR_RATIOS = [0.5,1,2]
__C.RPN_CHANNELS = 512
