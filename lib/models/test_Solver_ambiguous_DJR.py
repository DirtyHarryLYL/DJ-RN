from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose, Generate_action_object

import cv2
import cPickle as pickle
import numpy as np
import os
import sys
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + 'Ambiguous_HOI/' + image_id
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, detection):

    This_image = []

    im_orig, _ = get_blob(image_id) 
    blobs = {}

    blobs['H_num'] = 0
    blobs['H_boxes'] = np.empty((0, 5), dtype=np.float64)
    blobs['O_boxes'] = np.empty((0, 5), dtype=np.float64)
    blobs['gt_object'] = np.empty((0, 80), dtype=np.float64)
    blobs['pc']      = np.empty((0, 1228, 384), dtype=np.float64)
    blobs['att_2D_map'] = np.empty((0, 64, 64, 17), dtype=np.float32)
    blobs['sp']      = np.empty((0, 64, 64, 3), dtype=np.float32)
    blobs['smplx']   = np.empty((0, 89), dtype=np.float32)
    
    for i in range(len(Test_RCNN[image_id])):
        item = Test_RCNN[image_id][i]
        if item[1] == 'Human':
            blobs['H_num']   = 0
            blobs['H_boxes'], blobs['O_boxes'] = [], []
            blobs['gt_object'] = []
            blobs['pc'] = []
            blobs['att_2D_map'] = []
            blobs['sp'] = []
            blobs['smplx'] = []
            for j in range(len(Test_RCNN[image_id])):
                if i == j:
                    continue
                Object = Test_RCNN[image_id][j]
                blobs['H_num'] += 1
                blobs['H_boxes'].append(np.array([0, item[2][0], item[2][1], item[2][2], item[2][3]]).reshape(1,5))
                blobs['O_boxes'].append(np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 5))
                blobs['gt_object'].append(Generate_action_object(Object[4], 80))
                pat, att = Get_next_sp_with_pose(item[2], Object[2], item[6])
                blobs['sp'].append(pat)
                blobs['att_2D_map'].append(att)
                
                if os.path.exists(cfg.SMPLX_PATH + '/results/%s/%03d.pkl' % (image_id[:-4], item[-1])):
                    result = pickle.load(open(cfg.SMPLX_PATH + '/results/%s/%03d.pkl' % (image_id[:-4], item[-1])))
                    blobs['smplx'].append(np.concatenate([
                                        result['left_hand_pose'], result['right_hand_pose'],
                                        result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                        result['expression'], result['betas'],
                                    ], axis=1))
                    blobs['pc'].append(pickle.load(open(cfg.SMPLX_PATH + '/object_test/%s/human_%03d/%03d_feature.pkl' % (image_id[:-4], i, j), 'rb'))[None, ...])
                else:
                    blobs['smplx'].append(np.zeros((1, 85)))
                    blobs['pc'].append(np.zeros((1, 1228, 384)))

            if blobs['H_num'] == 0:
                continue
            
            blobs['H_boxes']    = np.concatenate(blobs['H_boxes'], axis=0)
            blobs['O_boxes']    = np.concatenate(blobs['O_boxes'], axis=0)
            blobs['gt_object']  = np.concatenate(blobs['gt_object'], axis=0)
            blobs['sp']         = np.concatenate(blobs['sp'], axis=0)
            blobs['att_2D_map'] = np.concatenate(blobs['att_2D_map'], axis=0)
            blobs['smplx']      = np.concatenate(blobs['smplx'], axis=0)
            blobs['pc']         = np.concatenate(blobs['pc'], axis=0)
            prediction_HO = net.test_image_HO(sess, im_orig, blobs)[0]

            for j in range(blobs['H_num']):
                Object = Test_RCNN[image_id][j]
                temp = []
                temp.append(item[1])           
                temp.append(Object[1])              
                temp.append(Object[0])              
                temp.append(prediction_HO[i])      
                temp.append(item[2])          
                temp.append(Object[2])             
                This_image.append(temp)
    detection[image_id] = This_image


def test_net(sess, net, Test_RCNN, output_dir):

    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for key in Test_RCNN.keys():

        _t['im_detect'].tic()

        im_detect(sess, net, key, Test_RCNN, detection)

        _t['im_detect'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1

    pickle.dump( detection, open( output_dir, "wb" ) )




