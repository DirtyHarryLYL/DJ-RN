from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose, Generate_action_object

import cPickle as pickle
import numpy as np
import os
import sys
import glob
import time
import cv2

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

obj_range = [
    (161, 170), (11, 24), (66, 76), (147, 160), (1, 10), 
    (55, 65), (187, 194), (568, 576), (32, 46), (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86), (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]

def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d

def im_detect(sess, net, image_id, Test_RCNN, keys, scores, bboxes, hdet, odet):
    im_file      = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im           = cv2.imread(im_file)
    im_orig      = im.astype(np.float32, copy=True)
    im_orig     -= cfg.PIXEL_MEANS
    im_orig      = im_orig.reshape(1, im_orig.shape[0], im_orig.shape[1], 3)
    blobs        = {}

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
                
                if os.path.exists(cfg.SMPLX_PATH + '/results/HICO_test2015_%08d/%03d.pkl' % (image_id, item[-1])):
                    result = pickle.load(open(cfg.SMPLX_PATH + '/results/HICO_test2015_%08d/%03d.pkl' % (image_id, item[-1])))
                    blobs['smplx'].append(np.concatenate([
                                        result['left_hand_pose'], result['right_hand_pose'],
                                        result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                        result['expression'], result['betas'],
                                    ], axis=1))
                    blobs['pc'].append(pickle.load(open(cfg.SMPLX_PATH + '/object_test/HICO_test2015_%08d/human_%03d/%03d_feature.pkl' % (image_id, i, j), 'rb'))[None, ...])
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
                classid = Object[4] - 1
                keys[classid].append(image_id)
                scores[classid].append(
                        prediction_HO[j][obj_range[classid][0] - 1:obj_range[classid][1]].reshape(1, -1) * \
                        getSigmoid(9, 1, 3, 0, item[5]) * \
                        getSigmoid(9, 1, 3, 0, Object[5]))
                hdet[classid].append(item[5])
                odet[classid].append(Object[5])
                hbox = np.array(item[2]).reshape(1, -1)
                obox = np.array(Object[2]).reshape(1, -1)
                bboxes[classid].append(np.concatenate([hbox, obox], axis=1))
    return

def test_net(sess, net, Test_RCNN, output_dir):
    
    np.random.seed(cfg.RNG_SEED)
    count = 0
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    keys, scores, bboxes, hdet, odet = [], [], [], [], []
    for i in range(80):
        keys.append([])
        scores.append([])
        bboxes.append([])
        hdet.append([])
        odet.append([])

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()
 
        image_id   = int(line[-9:-4])
        
        im_detect(sess, net, image_id, Test_RCNN, keys, scores, bboxes, hdet, odet)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1

    pickle.dump(keys, open(output_dir + 'keys.pkl', 'wb'))
    pickle.dump(scores, open(output_dir + 'scores.pkl', 'wb'))
    pickle.dump(bboxes, open(output_dir + 'bboxes.pkl', 'wb'))
    pickle.dump(hdet, open(output_dir + 'hdet.pkl', 'wb'))
    pickle.dump(odet, open(output_dir + 'odet.pkl', 'wb'))
