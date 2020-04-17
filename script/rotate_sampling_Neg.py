import os
import os.path as osp
import numpy as np
import pickle
import trimesh
import cv2
import matplotlib.pyplot as plt
import sympy, math
import pyrr
import configargparse
import torch
import smplx
from math import *
from human_body_prior.tools.model_loader import load_vposer
import argparse


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

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def get_joints(args,vertices):
    if (args.gender == 'neutral'):
        suffix = 'SMPLX_NEUTRAL.pkl'
    elif (args.gender == 'male'):
        suffix = 'SMPLX_MALE.pkl'
    else:
        suffix = 'SMPLX_FEMALE.pkl'
    smplx_path = args.smplx_path + suffix

    with open(smplx_path, 'rb') as smplx_file:
        model_data = pickle.load(smplx_file, encoding='latin1')
    
    data_struct = Struct(**model_data)
    j_regressor = to_tensor(to_np(
            data_struct.J_regressor), dtype=torch.float32)
    joints = vertices2joints(j_regressor, vertices) 
    return joints.numpy().reshape(-1,3)

def rotate_mul(verts, rotate):
    """
      verts [N,3]
      rotate [4,4]
    """
    rot = np.insert(verts, 3, values = 1, axis = 1)
    ret = np.dot(rot, rotate)
    return ret[:,:3]

def rotate(joints):
    s = [0,1,0]
    l = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
    x = s[0] / l
    y = s[1] / l
    z = s[2] / l
    
    a = 0
    b = 0
    c = 0

    u = x
    v = y
    w = z
    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w
    au = a * u
    av = a * v
    aw = a * w
    bu = b * u
    bv = b * v
    bw = b * w
    cu = c * u
    cv = c * v
    cw = c * w

    ansp = np.zeros((4,4))
    ans = 1000

    for i in range(1,1800):
      pi = acos(-1)
      ang = pi / 1800 * i
 
      v1 = joints[16]
      v2 = joints[17]
      
      sinA = sin(ang)
      cosA = cos(ang)
      costheta = cosA
      sintheta = sinA
      p = np.zeros((4,4))
      p[0][0] = uu + (vv + ww) * costheta
      p[0][1] = uv * (1 - costheta) + w * sintheta
      p[0][2] = uw * (1 - costheta) - v * sintheta
      p[0][3] = 0

      p[1][0] = uv * (1 - costheta) - w * sintheta
      p[1][1] = vv + (uu + ww) * costheta
      p[1][2] = vw * (1 - costheta) + u * sintheta
      p[1][3] = 0

      p[2][0] = uw * (1 - costheta) + v * sintheta
      p[2][1] = vw * (1 - costheta) - u * sintheta
      p[2][2] = ww + (uu + vv) * costheta
      p[2][3] = 0

      p[3][0] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta
      p[3][1] = (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta
      p[3][2] = (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta
      p[3][3] = 1

      v1 = v1.reshape(1,3)
      v2 = v2.reshape(1,3)
      rotv1 = np.dot(np.insert(v1, 3, values=1, axis=1),p)
      rotv2 = np.dot(np.insert(v2, 3, values=1, axis=1),p)

      if (abs(rotv1[0][2] - rotv2[0][2]) < ans):
        ans = abs(rotv1[0][2] - rotv2[0][2])
        ansp = p

    return ansp


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
        # item[5]: alphapose
        # item[6]: openpose index, -1 means none
        if not os.path.exists(os.path.join(args.res, 'results/HICO_train2015_%08d/%03d.pkl' % (key, i))):
                continue
        result = pickle.load(open(os.path.join(args.res, 'results/HICO_train2015_%08d/%03d.pkl' % (key, i)), 'rb'),encoding='latin1')
        hbox = item[2]
        obox = item[3]

        mesh       = os.path.join(args.res, 'meshes/HICO_train2015_%08d/%03d.obj' % (key, i))
        htri       = trimesh.load_mesh(mesh)
        vertice    = np.array(htri.vertices,dtype=np.float32)
        

        joints = get_joints(args,torch.FloatTensor(torch.from_numpy(vertice.reshape(1,-1,3))))
        ansp = rotate(joints)

        obj_file = os.path.join(args.obj_path, 'HICO_train2015_%08d/object_%03d.pkl' % (key, i))
        obj_vertice = np.array(pickle.load(open(obj_file, 'rb'),encoding='latin1'))
        
        path = os.path.join(args.save_path, 'HICO_train2015_%08d' % key)
        if (not os.path.exists(path)):
            os.makedirs(path)
        
        vertice = vertice[vertex_choice,:]

        pick_vertex = np.vstack((vertice,obj_vertice))
        pick_vertex = rotate_mul(pick_vertex, ansp)
        joints = rotate_mul(joints, ansp)
        pick_vertex = pick_vertex - joints[0]
        
        file_path = os.path.join(args.save_path, 'HICO_train2015_%08d/%03d.pkl' % (key, i))
        f = open(file_path,'wb') 
        pickle.dump(pick_vertex, f, protocol=2)
        
        obj_path_txt = obj_path_txt + file_path + '\n'

f = open('vertex_path_Neg.txt', 'w')
f.write(obj_path_txt)