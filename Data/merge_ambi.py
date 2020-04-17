import pickle
import shutil

ambi = pickle.load(open('ambi_names.pkl', 'rb'))
for key in ambi:
    if 'val2014' in key:
        shutil.copyfile('vcoco/val2014/' + key, 'Ambiguous_HOI/' + key)
    elif 'train2014' in key:
        shutil.copyfile('vcoco/train2014/' + key, 'Ambiguous_HOI/' + key)
    elif 'train2015' in key:
        shutil.copyfile('hico_20160224_det/image/train2015/' + key, 'Ambiguous_HOI/' + key)
    elif 'test2015' in key:
        shutil.copyfile('hico_20160224_det/image/test2015/' + key, 'Ambiguous_HOI/' + key)
