from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import cPickle as pickle
import random
from random import randint
import tensorflow as tf
import cv2
import os

from ult import config

print("*************data path:*************")
print(config.cfg.DATA_DIR)
print("************************************")

def bbox_trans(human_box_ori, object_box_ori, ratio, size = 64):
    human_box  = human_box_ori.copy()
    object_box = object_box_ori.copy()
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]    
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]    
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1] 
    if ratio == 'height':
        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width  - 1 - human_box[2]) / height
        human_box[3] = (size - 1)                  - size * (height - 1 - human_box[3]) / height
        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width  - 1 - object_box[2]) / height
        object_box[3] = (size - 1)                  - size * (height - 1 - object_box[3]) / height
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (InteractionPattern[2] + 1) / 2 
        human_box += [shift, 0 , shift, 0]
        object_box += [shift, 0 , shift, 0]
     
    else:

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1)                  - size * (width  - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width
        

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1)                  - size * (width  - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2 
        
        human_box = human_box + [0, shift, 0 , shift]
        object_box = object_box + [0, shift, 0 , shift]

 
    return np.round(human_box), np.round(object_box)

def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def Augmented_box(bbox, shape, augment=15, break_flag=True):

    thres_ = 0.7

    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
    box = box.astype(np.float64)
        
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        height = bbox[3] - bbox[1]
        width  = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen  = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
            count += 1
        if break_flag == True and time_count > 150:
            return box
            
    return box

def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,600)
    return action_

def draw_relation(human_pattern, joints, size = 64):

    joint_relation = [[1,3],[2,4],[0,1],[0,2],[0,17],[5,17],[6,17],[5,7],[6,8],[7,9],[8,10],[11,17],[12,17],[11,13],[12,14],[13,15],[14,16]]
    color = [0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))

    return skeleton

def get_skeleton(human_box, human_pose, human_pattern, num_joints = 17, size = 64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    xmap = np.tile(np.arange(64).reshape(1, -1), [64, 1]).astype(np.float32)
    ymap = np.tile(np.arange(64).reshape(-1, 1), [1, 64]).astype(np.float32)

    att_map_all = []
    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        att_map = 1 + np.sqrt((joint_x - xmap) ** 2, (joint_y - ymap) ** 2)
        att_map = 1. / att_map
        att_map /= np.sum(att_map)
        att_map = att_map.reshape((1, 64, 64, 1))
        att_map_all.append(att_map)
    att_map_all = np.concatenate(att_map_all, axis=3)

    return draw_relation(human_pattern, joints), att_map_all

def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64,64,2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1

    if human_pose != None and len(human_pose) == 51:
        skeleton, att_map = get_skeleton(human_box, human_pose, H, num_joints)
    else:
        skeleton = np.zeros((64,64,1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 0.05
        att_map = np.zeros((1, 64, 64, 17))

    Pattern = np.concatenate((Pattern, skeleton), axis=2).reshape(1, 64, 64, 3)

    return Pattern, att_map

def Get_Next_Instance_HO_Neg_HICO_3D(Trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select):

    im_file  = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, smplx, att_2D_map, pc = HO_Neg_HICO_3D(Trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['smplx']       = smplx
    blobs['gt_class_HO'] = action_HO
    blobs['att_2D_map']  = att_2D_map
    blobs['pc']          = pc

    return blobs

def HO_Neg_HICO_3D(Trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select):

    GT = Trainval_GT[image_id]
    GT_count = len(GT)
    GT_idx = list(np.random.choice(range(GT_count), Pos_augment))

    Human_augmented, Object_augmented, action_HO, Pattern, smplx, att_2D_map, pc = [], [], [], [], [], [], []
    
    for i in GT_idx:
        Human    = np.array(GT[i][2], dtype='float64')
        Object   = np.array(GT[i][3], dtype='float64')
        Human_augmented.append(np.array([0, Human[0], Human[1], Human[2], Human[3]], dtype='float64').reshape(1, -1))
        Object_augmented.append(np.array([0, Object[0], Object[1], Object[2], Object[3]], dtype='float64').reshape(1, -1))
        action_HO.append(Generate_action_HICO(GT[i][1]))
        pat_tmp, att_tmp = Get_next_sp_with_pose(Human, Object, GT[i][5])
        Pattern.append(pat_tmp)
        att_2D_map.append(att_tmp)
        if os.path.exists(config.cfg.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, GT[i][-1])):
            result = pickle.load(open(config.cfg.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, GT[i][-1])))
            smplx.append(np.concatenate([
                                            result['left_hand_pose'], result['right_hand_pose'],
                                            result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                            result['expression'], result['betas'],
                                        ], axis=1))
            pc.append(pickle.load(open(config.cfg.SMPLX_PATH + '/object_GT/HICO_train2015_%08d/%03d_feature.pkl' % i, 'rb'))[None, ...])
        else:
            smplx.append(np.zeros((1, 85)))
            pc.append(np.zeros((1, 1228, 384)))
        
    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for i in range(Trainval_Neg[image_id]):
                Neg = Trainval_Neg[image_id][i]
                Human_augmented.append(np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5))
                Object_augmented.append(np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5))
                action_HO.append(Generate_action_HICO([Neg[1]]))
                pat_tmp, att_tmp = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[4])
                Pattern.append(pat_tmp)
                att_2D_map.append(att_tmp)
                if os.path.exists(config.cfg.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, Neg[-1])):
                    result = pickle.load(open(config.cfg.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, Neg[-1])))
                    smplx.append(np.concatenate([
                                        result['left_hand_pose'], result['right_hand_pose'],
                                        result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                        result['expression'], result['betas'],
                                    ], axis=1))
                    pc.append(pickle.load(open(config.cfg.SMPLX_PATH + '/object_NEG/HICO_train2015_%08d/%03d_feature.pkl' % (image_id, i), 'rb'))[None, ...])
                else:
                    smplx.append(np.zeros((1, 85)))
                    pc.append(np.zeros((1, 1228, 384)))
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented.append(np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5))
                Object_augmented.append(np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5))
                action_HO.append(Generate_action_HICO([Neg[1]]))
                pat_tmp, att_tmp = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[4])
                Pattern.append(pat_tmp)
                att_2D_map.append(att_tmp)
                if os.path.exists(config.cfg.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, Neg[-1])):
                    result = pickle.load(open(config.cfg.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (image_id, Neg[-1])))
                    smplx.append(np.concatenate([
                                        result['left_hand_pose'], result['right_hand_pose'],
                                        result['leye_pose'], result['reye_pose'], result['jaw_pose'], result['body_pose'],
                                        result['expression'], result['betas'],
                                    ], axis=1))
                    pc.append(pickle.load(open(config.cfg.SMPLX_PATH + '/object_NEG/HICO_train2015_%08d/%03d_feature.pkl' % (image_id, i), 'rb'))[None, ...])
                else:
                    smplx.append(np.zeros((1, 85)))
                    pc.append(np.zeros((1, 1228, 384)))
    
    Pattern          = np.concatenate(Pattern, axis=0)
    Human_augmented  = np.concatenate(Human_augmented, axis=0)
    Object_augmented = np.concatenate(Object_augmented, axis=0)
    action_HO        = np.concatenate(action_HO, axis=0)
    smplx            = np.concatenate(smplx, axis=0)
    att_2D_map       = np.concatenate(att_2D_map, axis=0)
    pc               = np.concatenate(pc, axis=0)
    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, smplx, att_2D_map, pc

def Generate_action_object(idx, num_obj):
    action_obj = np.zeros([1, num_obj], dtype=np.float64)
    if isinstance(idx, int) or isinstance(idx, np.int32):
        action_obj[:, idx-1] = 1
    else:
        idx = np.array(list(idx))
        idx = idx - 1
        action_obj[:, list(idx)] = 1
    return action_obj
