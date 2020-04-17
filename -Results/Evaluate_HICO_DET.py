import cPickle as pickle
import numpy as np
import argparse
from HICO_DET_utils import calc_ap, obj_range, rare

def parse_args():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--file', dest='file',
            help='Detection file to evaluate',
            default='', type=str)
    args = parser.parse_args()
    return args

args = parse_args()

map = np.zeros(600)
mrec = np.zeros(600)
map_ko = np.zeros(600)
mrec_ko = np.zeros(600)

detection = pickle.load(open(args.file, 'rb'))

for obj_index in range(80):
    x, y = obj_range[obj_index]
    x -= 1
    
    ko_mask = []
    for hoi_id in range(x, y):
        gt_bbox = pickle.load(open('gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'))
        ko_mask += list(gt_bbox.keys())
    ko_mask = set(ko_mask)
    
    for hoi_index in range(x, y):
        bboxes = detection['bboxes'][hoi_index]
        scores = detection['scores'][hoi_index]
        keys   = detection['keys'][hoi_index]
        ap, _, ap_ko, _ = calc_ap(scores, bboxes, keys, hoi_index, ko_mask)
        map[hoi_index] = ap
        map_ko[hoi_index] = ap_ko

print('Default: ', np.mean(map))
print('Default rare: ', np.mean(map[rare > 1]))
print('Default non-rare: ', np.mean(map[rare < 1]))
print('Known object: ', np.mean(map_ko))
print('Known object, rare: ', np.mean(map_ko[rare > 1]))
print('Known object, non-rare: ', np.mean(map_ko[rare < 1]))