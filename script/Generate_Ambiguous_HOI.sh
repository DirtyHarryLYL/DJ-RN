#!/bin/bash

cd Data

python script/Download_data.py 1_aSwqUPCKHNSkx8WHIaHc3zjM0u50xPi Test_ambiguous.pkl
python script/Download_data.py 1w_h2eUZPgQFcGxIraPzmejJXE-mLo3m3 Ambiguous_HOI.tar
tar -xvf Ambiguous_HOI.tar
rm -rf Ambiguous_HOI.tar

URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip

wget -N $URL_2014_Train_images
wget -N $URL_2014_Val_images

if [! -d vcoco ];then
    mkdir vcoco
fi

unzip train2014.zip -d vcoco/
unzip val2014.zip -d vcoco/

rm train2014.zip
rm val2014.zip

python script/merge_for_ambi.py

cd ..