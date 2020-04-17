import pickle
import numpy as np
import json
import os
import argparse

body25_to_coco = [0, 15, 16, 17, 18, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

def calc_mse(pose1, pose2):
    tmp = pose1 - pose2
    tmp = np.sqrt(np.sum(tmp * tmp, axis=1))
    return np.mean(tmp)

def parse_args():
    parser = argparse.ArgumentParser(description='Assign pose to GT pkl')
    parser.add_argument('--pose', dest='pose',
            help='Path to your pose used to generate SMPLify-X result',
            default='', type=str)
    parser.add_argument('--res', dest='res',
            help='Path to your SMPLify-X result',
            default='', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
a = pickle.load(open('Data/Trainval_GT_HICO_with_pose.pkl', 'rb'), encoding='latin1')

data = {}
for item in a:
    if len(item) < 6:
        item.append(None)
    if item[0] not in data:
        data[item[0]] = []
    data[item[0]].append(item)

for key in data.keys():
    used = np.zeros(len(data[key])) - 1
    for item in data[key]:
        item.append(-1)
        # The structure of item here:
        # item[0]: image id
        # item[1]: hoi
        # item[2]: human bounding box
        # item[3]: object bounding box
        # item[4]: useless
        # item[5]: alphapose
        # item[6]: openpose index, -1 means none
    if os.path.exists(os.path.join(args.pose, 'HICO_train2015_%08d_keypoints.json' % key)):
        f = json.load(open(os.path.join(args.pose, 'HICO_train2015_%08d_keypoints.json' % key)))
        for i in range(len(f['people'])):
            tmp = np.array(f['people'][i]['pose_keypoints_2d'])
            if not os.path.exists(os.path.join(args.res, 'results/HICO_train2015_%08d/%03d.pkl' % (key, i))):
                continue
            body_pose = tmp.reshape(-1, 3)[body25_to_coco, :2]
            tmp_sum = np.sum(body_pose, axis=1)
            sel = np.where(tmp_sum > 0)[0]
            body_pose = body_pose[sel, :]
            
            for j in range(len(data[key])):
                if data[key][j][5] is not None:
                    alphapose = np.array(data[key][j][5]).reshape(-1, 3)[sel, :2]
                    dis = calc_mse(body_pose, alphapose)
                    if (used[j] < 0 and dis < 500) or (dis < used[j]):
                        used[j] = dis
                        data[key][j][-1] = i
pickle.dump(data, open('Data/Trainval_GT_HICO_with_idx.pkl', 'wb'), protocol=2)