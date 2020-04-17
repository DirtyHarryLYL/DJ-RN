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
from human_body_prior.tools.model_loader import load_vposer
import argparse

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

def get_order_obj():
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
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)]
    f = open('hico_list_hoi.txt','r')
    line = f.readline()
    line = f.readline()
    list_hoi = []
    list_hoi.append("None")
    line = f.readline()
    while line:
        tmp = line.strip('\n').split()
        list_hoi.append([tmp[1],tmp[2]])
        line = f.readline()

    obj_order_dict = {}
    order_obj_list = []
    order_obj_list.append(' ')
    for i in range(len(obj_range)):
        order_obj_list.append(list_hoi[obj_range[i][0]][0]) 
        obj_order_dict[order_obj_list[i+1]] = i + 1

    obj_para_dict = {}
    f = open('hico_obj_parameter.txt','r')
    line = f.readline()
    cnt = 0
    while line:
        cnt = cnt + 1
        tmp = line.strip('\n').split()
        tmp_dict = {}
        tmp_dict['ratio'] = float(tmp[1])
        tmp_dict['gamma_min'] = float(tmp[2])
        tmp_dict['gamma_max'] = float(tmp[3])
        obj_para_dict[tmp[0]] = tmp_dict
        line = f.readline()
    
    return list_hoi, order_obj_list, obj_para_dict

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

"""
Plot the orgin image and the 3D projection to 2D
""" 
def point_align_vis(result, obox, mesh, img):
    img = cv2.imread(img)[:, :, ::-1].astype(np.float32) / 255.   
    rotation = result['camera_rotation'][0, :, :]
    camera_trans = result['camera_translation']
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = rotation
    camera_transform[:3, 3]  = camera_trans
    camera_mat = np.zeros((2, 2))
    camera_mat[0, 0] = 5000.
    camera_mat[1, 1] = 5000
    vert = []
    with open(mesh) as f:
        while True:
            line = f.readline().split()
            if line[0] == 'v':
                vert.append(np.array([float(line[1]), float(line[2]), float(line[3])]))
            else:
                break
    vert = np.array(vert)
    camera_center = np.array([img.shape[1], img.shape[0]]) * 0.5
    camera_center = camera_center.astype(np.int32)
    homog_coord = np.ones(list(vert.shape[:-1]) + [1])
    points_h = np.concatenate([vert, homog_coord], axis=-1)
    for i in range(points_h.shape[0]):
        point = points_h[i]
        point[1] *= -1
        projected = np.matmul(camera_transform, point)
        img_point = projected[:2] / projected[2]
        img_point = np.matmul(camera_mat, img_point)
        img_point = img_point + camera_center
        img_point = img_point.astype(np.int32)
        img = cv2.circle(img, (img_point[0], img_point[1]), 5, (0, 1, 0), -1)
    img = cv2.rectangle(img, (obox[0], obox[1]), (obox[2], obox[3]),(1, 0, 0), 2)
    plt.imshow(img)

def icosahedron():
    """Construct a 20-sided polyhedron"""
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1),
        (11, 7, 6), (11, 8, 7), (11, 9, 8), (11, 10, 9), (11, 6, 10),
        (1, 6, 2), (2, 7, 3), (3, 8, 4), (4, 9, 5), (5, 10, 1),
        (6, 7, 2), (7, 8, 3), (8, 9, 4), (9, 10, 5), (10, 6, 1),
    ]
    verts = [
        [0.000, 0.000, 1.000],  [0.894, 0.000, 0.447], [0.276, 0.851, 0.447],
        [-0.724, 0.526, 0.447], [-0.724, -0.526, 0.447], [0.276, -0.851, 0.447],
        [0.724, 0.526, -0.447], [-0.276, 0.851, -0.447], [-0.894, 0.000, -0.447],
        [-0.276, -0.851, -0.447], [0.724, -0.526, -0.447], [0.000, 0.000, -1.000],
    ]
    return verts, faces
def subdivide(verts, faces):
    """Subdivide each triangle into four triangles, pushing verts to the unit sphere"""
    triangles = len(faces)
    for faceIndex in range(triangles):

        # Create three new verts at the midpoints of each edge:
        face = faces[faceIndex]
        a, b, c = np.float32([verts[vertIndex] for vertIndex in face])
        verts.append(pyrr.vector.normalize(a + b))
        verts.append(pyrr.vector.normalize(b + c))
        verts.append(pyrr.vector.normalize(a + c))

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i + 1, i + 2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[faceIndex] = (k, j, face[2])
    return verts, faces

def cal_r_rule(d, r_ratio):
    dis = np.sqrt(np.sum(d * d))
    r = dis * r_ratio
    return r
def get_param(result, hbox, obox, htri, img, radius=None, gamma_min=None, gamma_max=None):    
    focal_length = 5000
    root1  = pickle.load(open('equation-root1.pkl', 'rb'), encoding='latin1')
    root1r = pickle.load(open('equation-root1r.pkl', 'rb'), encoding='latin1')
    rotation = result['camera_rotation'][0, :, :]
    camera_transl = result['camera_translation']
    camera_transform = np.eye(4)
    camera_transform[:3, :3] = rotation
    camera_transform[:3, 3]  = camera_transl
    camera_mat = np.eye(2).astype(np.float32) * focal_length

    vert = np.array(htri.vertices)

    img    = cv2.imread(img)[:, :, ::-1].astype(np.float32) / 255.
    camera_center = np.array([img.shape[1], img.shape[0]]) * 0.5
    camera_center = camera_center.astype(np.int32)
    
    hbox[0] -= camera_center[0]
    hbox[1] -= camera_center[1]
    hbox[2] -= camera_center[0]
    hbox[3] -= camera_center[1]
    obox[0] -= camera_center[0]
    obox[1] -= camera_center[1]
    obox[2] -= camera_center[0]
    obox[3] -= camera_center[1]

    x_mid = (obox[0] + obox[2]) / 2
    y1, y2 = obox[1], obox[3]

    t1, t2, t3 = camera_transl[0, 0], camera_transl[0, 1], camera_transl[0, 2]

    ly1_x = [x_mid / focal_length, x_mid * t3 / focal_length - t1]
    ly1_y = [-y1 / focal_length, -y1 * t3 / focal_length + t2]
    ly2_x = [x_mid / focal_length, x_mid * t3 / focal_length - t1]
    ly2_y = [-y2 / focal_length, -y2 * t3 / focal_length + t2]
    vec_1 = np.array([ly1_x[0], ly1_y[0], 1])
    vec_2 = np.array([ly2_x[0], ly2_y[0], 1])
    top = np.sum(vec_1 * vec_2)
    bottom = np.sqrt(np.sum(vec_1 * vec_1)) * np.sqrt(np.sum(vec_2 * vec_2))
    theta = np.arccos(top / bottom)

    _t1 = t1
    _t2 = t2
    _t3 = t3
    _x_mid = x_mid
    _theta = theta
    _focal_length = focal_length
    x = sympy.Symbol('x', real=True)
    y = sympy.Symbol('y', real=True)
    z = sympy.Symbol('z', real=True)
    t1 = sympy.Symbol('t1', real=True)
    t2 = sympy.Symbol('t2', real=True)
    t3 = sympy.Symbol('t3', real=True)
    x_mid = sympy.Symbol('x_mid', real=True)
    theta = sympy.Symbol('theta', real=True)
    focal_length = sympy.Symbol('focal_length', real=True)
    vec_20 = sympy.Symbol('vec_20', real=True)
    vec_21 = sympy.Symbol('vec_21', real=True)
    vec_22 = sympy.Symbol('vec_22', real=True)
    r = sympy.Symbol('r', real=True)

    maxz = np.max(vert[:, 2]) * gamma_max
    minz = np.min(vert[:, 2]) * gamma_min
    
    
    if radius is not None:
        value = {t1: _t1, t2: _t2, t3: _t3, x_mid: _x_mid, theta: _theta, focal_length: _focal_length, vec_20: vec_2[0],
                 vec_21: vec_2[1], vec_22: vec_2[2], r: radius}
        for i in range(4):
            ansx = root1[i][0].evalf(subs=value)
            ansy = root1[i][1].evalf(subs=value)
            ansz = root1[i][2].evalf(subs=value)
            y2D = (-ansy + _t2) / (ansz + _t3) * _focal_length
            x2D = (-ansx + _t1) / (ansz + _t3) * _focal_length
            if (((y2D >= obox[1]) and (y2D <= obox[3])) or ((y2D <= obox[1]) and (y2D >= obox[3]))):
                idx = i
    
        ansx = root1[idx][0].evalf(subs=value)
        ansy = root1[idx][1].evalf(subs=value)
        ansz = root1[idx][2].evalf(subs=value)
        
        # Deal with the condition of the object goes beyond a certain range
        # print(ansz, minz, maxz)
        if (ansz > maxz or ansz < minz):          
            if (ansz > maxz): ansz = maxz
            if (ansz < minz): ansz = minz
            value = {t1: _t1, t2: _t2, t3: _t3, x_mid: _x_mid, theta: _theta, focal_length: _focal_length, vec_20: vec_2[0],
                 vec_21: vec_2[1], vec_22: vec_2[2], z: ansz}
            for i in range(2):
                ansx = root1r[i][0].evalf(subs=value)
                ansy = root1r[i][1].evalf(subs=value)
                y2D = (-ansy + _t2) / (ansz + _t3) * _focal_length
                x2D = (ansx + _t1) / (ansz + _t3) * _focal_length
                if (((y2D >= obox[1]) and (y2D <= obox[3])) or ((y2D <= obox[1]) and (y2D >= obox[3]))):
                    idx = i
            ansx = root1r[idx][0].evalf(subs=value)
            ansy = root1r[idx][1].evalf(subs=value)
            radius = root1r[idx][2].evalf(subs=value)

        point = [float(ansx), float(ansy), float(ansz)]
        point = np.append(point, 1)
        ansr = radius
    else:
        R = cal_r_rule(vert[9448] - vert[9929], 1)
        left = R / 10
        right = R * 100
        flag, ansr, idx, flag2, flag3, tot = 0, 0, -1, 0, 0, 0
        while (flag == 0 and tot < 15):
            R = (left + right) / 2
            tot = tot + 1
            value = {t1: _t1, t2: _t2, t3: _t3, x_mid: _x_mid, theta: _theta, focal_length: _focal_length, vec_20: vec_2[0],
                     vec_21: vec_2[1], vec_22: vec_2[2], r: R}
            if (flag2 == 0):
                flag2 = 1
                for i in range(4):
                    ansx = root1[i][0].evalf(subs=value)
                    ansy = root1[i][1].evalf(subs=value)
                    ansz = root1[i][2].evalf(subs=value)
                    y2D = (-ansy + _t2) / (ansz + _t3) * _focal_length
                    x2D = (ansx + _t1) / (ansz + _t3) * _focal_length
                    if (math.isnan(y2D)):
                        flag3 = 1
                        break
                    if (((y2D >= obox[1]) and (y2D <= obox[3])) or ((y2D <= obox[1]) and (y2D >= obox[3]))):
                        idx = i
            if (flag3 == 1):
                break
            ansx = root1[idx][0].evalf(subs=value)
            ansy = root1[idx][1].evalf(subs=value)
            ansz = root1[idx][2].evalf(subs=value)
    
            point = [float(ansx), float(ansy), float(ansz)]
            point = np.append(point, 1)
    
            if (point[2] < minz):
                left = R
            elif (point[2] > maxz):
                right = R
            elif (point[2] >= minz and point[2] <= maxz):
                flag = 1
                ansr = float(R)
    
    # print(ansx,ansy,ansz, ansr)
    verts, faces = icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    for i in range(len(verts)):
        verts[i][0] *= ansr
        verts[i][1] *= ansr
        verts[i][2] *= ansr
        verts[i][0] += point[0]
        verts[i][1] += point[1]
        verts[i][2] += point[2]
    otri = trimesh.Trimesh(vertices=verts, faces=faces)
    return otri, verts

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