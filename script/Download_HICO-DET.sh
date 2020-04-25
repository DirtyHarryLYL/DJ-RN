#!/bin/bash

# URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz
# new url hico-det from their website: https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk

# wget -N $URL_HICO_DET -P Data/
python script/Download_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

python script/Download_data.py 1L_AEJ3sWYbhZMqwYqSXLHohLnS5IW3vz Data.tar
tar -xvf Data.tar
rm -rf Data.tar

