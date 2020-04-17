#!/bin/bash

URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz

wget -N $URL_HICO_DET -P Data/
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

python script/Download_data.py 1L_AEJ3sWYbhZMqwYqSXLHohLnS5IW3vz Data.tar
tar -xvf Data.tar
rm -rf Data.tar

