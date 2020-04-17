import cPickle as pickle
import numpy as np
import os
import argparse
from HICO_DET_utils import calc_ap, obj_range, rare, getSigmoid, hoi_no_inter_all
from HICO_DET_utils import obj_range, getSigmoid, hoi_no_inter_all

def parse_args():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--model', dest='model',
            help='Select model to generate',
            default='', type=str)
    args = parser.parse_args()
    return args

args = parse_args()

scores   = pickle.load(open(os.path.join(args.model, 'scores.pkl'), 'rb'))
hdet     = pickle.load(open(os.path.join(args.model, 'hdet.pkl'), 'rb'))
odet     = pickle.load(open(os.path.join(args.model, 'odet.pkl'), 'rb'))
keys     = pickle.load(open(os.path.join(args.model, 'keys.pkl'), 'rb'))
bboxes   = pickle.load(open(os.path.join(args.model, 'bboxes.pkl'), 'rb'))
pos      = pickle.load(open('pos.pkl', 'rb'))
neg      = pickle.load(open('neg.pkl', 'rb'))

for obj_index in range(80):
    scores[obj_index] = np.concatenate(scores[obj_index], axis=0)
    bboxes[obj_index] = np.concatenate(bboxes[obj_index], axis=0)
    keys[obj_index]   = np.array(keys[obj_index])
    hdet[obj_index]   = np.array(hdet[obj_index])
    odet[obj_index]   = np.array(odet[obj_index])
    pos[obj_index]    = np.array(pos[obj_index])
    neg[obj_index]    = np.array(neg[obj_index])
    
hthresh, othresh, athresh, bthresh = pickle.load(open('generation_args.pkl', 'rb'))

detection = {}
detection['bboxes'] = []
detection['scores'] = []
detection['keys']    = []

for i in range(600):
    detection['keys'].append([])
    detection['scores'].append([])
    detection['bboxes'].append([])


for obj_index in range(80):
    x, y = obj_range[obj_index]
    x -= 1
    
    inter_det_mask = (hdet[obj_index] > hthresh[x]) * (odet[obj_index] > othresh[x])
    no_inter_det_mask = (hdet[obj_index] > hthresh[y-1]) * (odet[obj_index] > othresh[y-1])

    for hoi_index in range(x, y):
        at, bt = athresh[hoi_index], bthresh[hoi_index]
        if hoi_index + 1 in hoi_no_inter_all:
            nis_mask = 1 - (pos[obj_index] > at) * (neg[obj_index] < bt)
            mask   = no_inter_det_mask * nis_mask
        else:
            nis_mask = 1 - (pos[obj_index] < at) * (neg[obj_index] > bt)
            mask = inter_det_mask * nis_mask
        select        = np.where(mask > 0)[0]
        if len(select) > 0:
            detection['scores'][hoi_index] = scores[obj_index][select, hoi_index - x]
            detection['bboxes'][hoi_index] = bboxes[obj_index][select]
            detection['keys'][hoi_index]   = keys[obj_index][select]

pickle.dump(detection, open('Detection_' + args.model + '.pkl', 'wb'))
        