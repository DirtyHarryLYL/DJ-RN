import os
import os.path as osp
import numpy as np
import pickle
import trimesh
import torch
import argparse
from generate_utils import get_order_obj, get_joints, get_param, rotate_mul, rotate


def parse_args():
    parser = argparse.ArgumentParser(description='Rotate and downsampling on vertexs')
    
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

    parser.add_argument('--obj_path', dest='obj_path',
            help='Path to get your obj 3D mesh',
            default='', type=str)

    parser.add_argument('--save_path', dest='save_path',
            help='Path to save the picking vertexs pkl',
            default='', type=str)

    args = parser.parse_args()
    return args

args = parse_args()

vertex_choice = pickle.load(open('remaining_vertexs-part.pkl','rb'),encoding='latin1')
vertex_choice = np.array(vertex_choice)
vertex_choice = vertex_choice[:,0]

data = pickle.load(open('../Data/Trainval_Neg_HICO_with_idx.pkl', 'rb'), encoding='latin1')

obj_path_txt = ''

for key in data.keys():
    for i, item in enumerate(data[key]):
        # The structure of item here:
        # item[0]: image id
        # item[1]: hoi
        # item[2]: human bounding box
        # item[3]: object bounding box
        # item[4]: useless
        # item[5]: useless
        # item[6]: useless
        # item[7]: alphapose
        # item[8]: openpose index, -1 means none
        if not os.path.exists(os.path.join(args.res, 'results/HICO_train2015_%08d/%03d.pkl' % (key, item[8]))):
            continue
        result = pickle.load(open(os.path.join(args.res, 'results/HICO_train2015_%08d/%03d.pkl' % (key, item[8])), 'rb'),encoding='latin1')
        hbox = item[2]
        obox = item[3]

        mesh       = os.path.join(args.res, 'meshes/HICO_train2015_%08d/%03d.obj' % (key, item[8]))
        htri       = trimesh.load_mesh(mesh)
        vertice    = np.array(htri.vertices,dtype=np.float32)
        

        joints = get_joints(args,torch.FloatTensor(torch.from_numpy(vertice.reshape(1,-1,3))))
        ansp = rotate(joints - joints[0])

        obj_file = os.path.join(args.obj_path, 'HICO_train2015_%08d/object_%03d.pkl' % (key, i))
        obj_vertice = np.array(pickle.load(open(obj_file, 'rb'),encoding='latin1'))
        
        path = os.path.join(args.save_path, 'HICO_train2015_%08d' % key)
        if (not os.path.exists(path)):
            os.makedirs(path)
        
        vertice = vertice[vertex_choice,:]

        pick_vertex = np.vstack((vertice,obj_vertice))
        pick_vertex = pick_vertex - joints[0]
        pick_vertex = rotate_mul(pick_vertex, ansp)
        
        file_path = os.path.join(args.save_path, 'HICO_train2015_%08d/%03d.pkl' % (key, i))
        f = open(file_path,'wb') 
        pickle.dump(pick_vertex, f, protocol=2)
        
        obj_path_txt = obj_path_txt + file_path + '\n'

f = open('vertex_path_Neg.txt', 'w')
f.write(obj_path_txt)
