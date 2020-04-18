from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np
import os
import sys
import argparse

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
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]

ambiguous = [522, 280, 405, 593, 51, 66, 58, 339, 206, 255, 485, 44, 167, 22, 418, 399, 555, 274, 318, 496, 84, 426, 427, 411, 490, 165, 235, 107, 100, 65, 322, 127, 345, 586, 211, 525, 80, 378, 114, 129, 344, 350, 126, 579, 135, 531, 33, 188, 163, 72, 210, 184, 180, 158, 548, 239, 493, 472, 62, 488, 113, 189, 24, 2, 247, 121, 368, 225, 334, 92, 59, 327, 161, 104, 343, 123, 430, 69, 4, 415, 74, 105, 28, 136, 550, 315, 372]

def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d

def iou(bb1, bb2):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    
    
    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    
    
    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)

def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    det   = det.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)

def calc_ap(anno, scores, bboxes, keys, hoi_id, begin):
    if hoi_id not in ambiguous:
        return np.nan, np.nan
    score = scores[:, hoi_id - begin]
    hit = []
    idx = np.argsort(score)[::-1]
    for i in range(len(ambiguous)):
        if ambiguous[i] == hoi_id:
            gt_bbox = anno[i]['data']
    npos = 0
    used = {}
    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        return 0, 0
    for i in range(len(idx)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key  = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k    = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0.)
            else:
                hit.append(1.)
                used[key].add(k)
        else:
            hit.append(0.)
    bottom = np.array(range(len(hit)), dtype=np.float32) + 1.
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    return ap, np.max(rec)

def Generate_ambi_detection(ambi, ambi_DIR):
    anno = pickle.load(open('gt_ambiguous_hoi.pkl', 'rb'))
    keys, scores, bboxes, hdet, odet = [], [], [], [], []

    for i in range(80):
        keys.append([])
        scores.append([])
        bboxes.append([])
        hdet.append([])
        odet.append([])

    for key, value in ambi.items():
        for element in value:
            classid = element[2] - 1
            keys[classid].append(key)
            scores[classid].append(element[3][obj_range[classid][0]-1:obj_range[classid][1]].reshape(1, -1) * element[4] * element[5])
            hbox = np.array(element[0]).reshape(1, -1)
            obox = np.array(element[1]).reshape(1, -1)
            bboxes[classid].append(np.concatenate([hbox, obox], axis=1))
            hdet[classid].append(element[4])
            odet[classid].append(element[5])
    print("Preparation finished")
    
    map  = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end   = obj_range[i][1]
        if len(scores[i]) == 0:
            map[begin:end]  = np.nan
            mrec[begin:end] = np.nan
            continue
        scores[i] = np.concatenate(scores[i], axis=0)
        bboxes[i] = np.concatenate(bboxes[i], axis=0)
        keys[i]   = np.array(keys[i])
        hdet[i]   = np.array(hdet[i])
        odet[i]   = np.array(odet[i])
        mask = (hdet[i] > 0.0) * (odet[i] > 0.0)
        select = np.where(mask > 0)
        for hoi_id in range(begin, end):
            map[hoi_id], mrec[hoi_id] = calc_ap(anno, scores[i][select], bboxes[i][select], keys[i][select], hoi_id, begin)
            if hoi_id not in ambiguous:
                continue
    pickle.dump({'ap':map, 'rec':mrec}, open(ambi_DIR + '/eval_def_res.pkl', 'wb'))

    f = open(ambi_DIR + '/eval_result.txt', 'w')
    f.write("mAP: %.4f, mRec: %.4f" % (float(np.nanmean(map)), float(np.nanmean(mrec))))
    f.close()

    print("mAP: %.4f, mRec: %.4f" % (float(np.nanmean(map)), float(np.nanmean(mrec))))



def main():
    output_file = sys.argv[1]
    ambi_dir = sys.argv[2]
    
    ambi = pickle.load(open(output_file, "rb"))
    print("Pickle loaded")
    if not os.path.exists(ambi_dir):
        os.mkdir(ambi_dir)
    Generate_ambi_detection(ambi, ambi_dir)

if __name__ == '__main__':
    main()
    