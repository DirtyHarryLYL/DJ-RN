#!/bin/bash

python script/Download_data.py 1ELTODP3Rgv09v00ft-H4DeEmqR6ZLMnJ ./-Results.tar
tar -xvf ./-Results.tar
rm -rf ./-Results.tar

python script/Download_data.py 10nUe9ruH7ZPTEh8WuV7lRCrRMCGFocKz ./-Results/400000_DJR_ambiguous.pkl.tar.gz
cd ./-Results
tar -xvzf 400000_DJR_ambiguous.pkl.tar.gz
rm -rf 400000_DJR_ambiguous.pkl.tar.gz
cd ..