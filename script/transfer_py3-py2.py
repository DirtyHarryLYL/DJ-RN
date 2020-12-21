import os
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on HICO')

    parser.add_argument('--res', dest='res',
            help='Path to your SMPLify-X result',
            default='smplx_res/', type=str)

    args = parser.parse_args()
    return args

args = parse_args()
result_path = args.res + 'results'
res_list = os.listdir(result_path)
for i, suffix in enumerate(res_list):
    path =  os.path.join(result_path,suffix)
    _list = os.listdir(path)
    for j, _suffix in enumerate(_list):
        f = os.path.join(path,_suffix)
        tmp = pickle.load(open(f,'rb'),encoding='latin1')

        with open(f, "wb") as out:
            pickle.dump(tmp, out , protocol=2)
       
    

