import os
import os.path as osp
import numpy as np
import pickle
import trimesh
import torch
import argparse
from generate_utils import get_order_obj, get_joints, get_param

def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on HICO')
    
    parser.add_argument('--smplx_path', dest='smplx_path',
            help='Path to your SMPLX model',
            default='', type=str)

    parser.add_argument('--gender', dest='gender',
            help='Use gender neutral or gender specific SMPLX' + 'model',
            default='male',choices=['neutral', 'male', 'female'],
            type=str )

    parser.add_argument('--res', dest='res',
            help='Path to your SMPLify-X result',
            default='', type=str)

    parser.add_argument('--img_path', dest='img_path',
            help='Path to your image',
            default='', type=str)

    parser.add_argument('--save_obj_path', dest='save_obj_path',
            help='Path to save generated obj',
            default='', type=str)

    args = parser.parse_args()
    return args

args = parse_args()
list_hoi, order_obj_list, obj_para_dict = get_order_obj()
data = pickle.load(open('../Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_idx.pkl', 'rb'), encoding='latin1')
for key in data.keys():
    img = os.path.join(args.img_path, 'HICO_test2015_%08d.jpg' % (key))
    for i, item in enumerate(data[key]):
        if (item[1] == 'Object'):
            continue
        for j, item2 in enumerate(data[key]):
            if (j == i): continue
            # The structure of item here:
            # item[0]: image id
            # item[1]: 'Human' of 'Object'
            # item[2]: human bounding box
            # item[3]: nan
            # item[4]: object category
            # item[5]: object detection score
            # item[6]: alphapose
            # item[7]: openpose index, -1 means none

            if not os.path.exists(os.path.join(args.res, 'results/HICO_test2015_%08d/%03d.pkl' % (key, i))):
                    continue

            result = pickle.load(open(os.path.join(args.res, 'results/HICO_test2015_%08d/%03d.pkl' % (key, i)), 'rb'))
            hbox = item[2]
            obox = item2[2]

            obj_name = order_obj_list[item2[4]] 


            mesh       = os.path.join(args.res, 'meshes/HICO_test2015_%08d/%03d.obj' % (key, i))
            htri       = trimesh.load_mesh(mesh)
            vertice    = np.array(htri.vertices,dtype=np.float32)
            
            joints = get_joints(args,torch.FloatTensor(torch.from_numpy(vertice.reshape(1,-1,3))))
            shoulder_len = np.linalg.norm(joints[16] - joints[17])
            
            radius = obj_para_dict[obj_name]['ratio'] * shoulder_len
            gamma_min = obj_para_dict[obj_name]['gamma_min']
            gamma_max = obj_para_dict[obj_name]['gamma_max']  
            
            otri, obj_vertexs = get_param(result, hbox, obox, htri, img, radius, gamma_min, gamma_max)

            path = os.path.join(args.save_obj_path, 'HICO_test2015_%08d' % key)
            if (not os.path.exists(path)):
                os.makedirs(path)
            path = os.path.join(args.save_obj_path, 'HICO_test2015_%08d/human_%03d' % (key,i))
            if (not os.path.exists(path)):
                os.makedirs(path)

            file_path = os.path.join(args.save_obj_path, 'HICO_test2015_%08d/human_%03d/object_%03d.pkl' % (key, i, j))
            f = open(file_path,'wb') 
            pickle.dump(obj_vertexs, f, protocol=2)    

            tmp = otri.export(os.path.join(args.save_obj_path, 'HICO_test2015_%08d/human_%03d/object_%03d.obj' % (key, i, j)))
            alltri = otri + htri
            tmp = alltri.export(os.path.join(args.save_obj_path, 'HICO_test2015_%08d/human_%03d/human_object_%03d.obj' % (key, i, j)))