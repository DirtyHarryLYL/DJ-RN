#!/bin/bash

python script/Download_data.py 1AUBiHClph9R8e6bPzuZTebuwWSvwjZR8 Weights/DJR.tar
cd Weights
tar -xvf DJR.tar
rm -rf DJR.tar
cd ..