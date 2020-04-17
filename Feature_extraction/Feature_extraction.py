import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import cPickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_hico', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--num_point', type=int, default=1228, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--input_list', default='./', help='Path list of your point cloud files [default: ./pc_list.txt]')
FLAGS = parser.parse_args()


NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
BATCH_SIZE = 1
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

MAX_NUM_POINT = 1228
NUM_CLASSES = 600

HOSTNAME = socket.gethostname()


def evaluate():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        feat = MODEL.get_model(pointclouds_pl, is_training_pl)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)

    ops = {'pointclouds_pl': pointclouds_pl,
           'is_training_pl': is_training_pl,
           'feat': feat}

    eval_one_epoch(sess, ops)

   
def eval_one_epoch(sess, ops):
    is_training = False
    input_list = None
    with open(FLAGS.input_list, 'r') as f:
        input_list = f.readlines()
    
    for fn in range(len(input_list)):
        current_data = pickle.load(open(fn, 'rb'))
        current_data = current_data[None, :NUM_POINT, :]
        
            
        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['is_training_pl']: is_training}
        feat = sess.run([ops['feat']], feed_dict=feed_dict)
        pickle.dump(feat, open(fn[:-4] + '_feature.pkl', 'wb'))

with tf.Graph().as_default():
    evaluate()