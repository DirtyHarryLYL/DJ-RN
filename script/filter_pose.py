import os
import numpy as np
import json
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Filter pose')
    parser.add_argument('--origin_pose', dest='ori',
            help='Path to your generated openpose directory',
            default='', type=str)
    parser.add_argument('--filtered_pose', dest='fil',
            help='Path to the filtered openpose',
            default='', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
if not os.path.exists(args.fil):
    os.mkdir(args.fil)

for line in glob.iglob(args.ori + '/*.json'):
    image_name = line[len(args.ori):-15]
    f = json.load(open(line))
    preserve = []
    for j in range(len(f['people'])):
        tmp = np.array(f['people'][j]['pose_keypoints_2d'])
        if tmp[2] != 0 and tmp[5] != 0 and tmp[8] != 0 and tmp[17] != 0 and tmp[26] != 0 and tmp[29] != 0 and tmp[38] != 0:
            preserve.append(f['people'][j])
    if len(preserve) > 0:
        f['people'] = preserve
        json.dump(f, open(os.path.join(args.fil, image_name + '_keypoints.json'), 'w'))
    