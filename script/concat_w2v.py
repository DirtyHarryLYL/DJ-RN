import numpy as np
import pickle
import os
import argparse

obj_range = [
    (161, 170), (11, 24),   (66, 76),   (147, 160), (1, 10), 
    (55, 65),   (187, 194), (568, 576), (32, 46),   (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86),   (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92),   (93, 96),   (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]
mapping = np.zeros(600)
for i in range(80):
    x, y = obj_range[i]
    mapping[x - 1:y] = i

part_vec = pickle.load(open('part-w2v.pkl', 'rb'))
obj_vec  = pickle.load(open('obj-w2v.pkl', 'rb'))
a = np.array(pickle.load(open('remaining_vertexs-part.pkl', 'rb')))[:, 1]
part_holder = part_vec[a]

def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on HICO')
    
    parser.add_argument('--SMPLX_PATH', dest='SMPLX_PATH',
            help='Path to your SMPLX results',
            default='', type=str)

    parser.add_argument('--GT', dest='GT',
            help='Path to the GT pkl',
            default='',
            type=str )

    parser.add_argument('--Neg', dest='Neg',
            help='Path to the Neg pkl',
            default='',
            type=str )

    parser.add_argument('--Test', dest='Test',
            help='Path to the Test pkl',
            default='',
            type=str )

    args = parser.parse_args()
    return args

args = parse_args()
Trainval_GT = pickle.load(open(args.GT, 'rb'))
Trainval_Neg = pickle.load(open(args.Neg, 'rb'))
Test_RCNN = pickle.load(open(args.Test, 'rb'))

for key in Trainval_GT:
    for i in range(len(Trainval_GT[key])):
        hoi_class = Trainval_GT[key][i][1][0]
        cur_obj   = mapping[hoi_class]
        obj_holder = np.tile(obj_vec[cur_obj], (312, 1))
        holder = np.concatenate([part_holder, obj_holder], axis=0)
        if os.path.exists(args.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (key, Trainval_GT[key][i][-1])):
            pc = pickle.load(open(args.SMPLX_PATH + '/object_GT/HICO_train2015_%08d/%03d_feature.pkl' % (key, i), 'rb'))
            pc = np.concatenate([pc, holder], axis=1)
            pickle.dump(pc, open(args.SMPLX_PATH + '/object_GT/HICO_train2015_%08d/%03d_feature.pkl' % (key, i), 'wb'))

for key in Trainval_Neg:
    for i in range(len(Trainval_Neg[key])): 
        hoi_class = Trainval_Neg[key][i][1]
        cur_obj   = mapping[hoi_class]
        obj_holder = np.tile(obj_vec[cur_obj], (312, 1))
        holder = np.concatenate([part_holder, obj_holder], axis=0)
        if os.path.exists(args.SMPLX_PATH + '/results/HICO_train2015_%08d/%03d.pkl' % (key, Trainval_Neg[key][i][-1])):
            pc = pickle.load(open(args.SMPLX_PATH + '/object_NEG/HICO_train2015_%08d/%03d_feature.pkl' % (key, i), 'rb'))
            pc = np.concatenate([pc, holder], axis=1)
            pickle.dump(pc, open(args.SMPLX_PATH + '/object_NEG/HICO_train2015_%08d/%03d_feature.pkl' % (key, i), 'wb'))

for image_id in Test_RCNN:
    for i in range(len(Test_RCNN[image_id])):
        item = Test_RCNN[image_id][i]
        if item[1] == 'Human':
            for j in range(len(Test_RCNN[image_id])):
                if i == j:
                    continue
                Object = Test_RCNN[image_id][j]
                cur_obj = Object[4] - 1
                obj_holder = np.tile(obj_vec[cur_obj], (312, 1))
                holder = np.concatenate([part_holder, obj_holder], axis=0)
                if os.path.exists(args.SMPLX_PATH + '/results/HICO_test2015_%08d/%03d.pkl' % (image_id, item[-1])):
                    pc = pickle.load(open(args.SMPLX_PATH + '/object_test/HICO_test2015_%08d/human_%03d/%03d_feature.pkl' % (image_id, i, j), 'rb'))
                    pc = np.concatenate([pc, holder], axis=1)
                    pickle.dump(pc, open(args.SMPLX_PATH + '/object_test/HICO_test2015_%08d/human_%03d/%03d_feature.pkl' % (image_id, i, j), 'rb'))
