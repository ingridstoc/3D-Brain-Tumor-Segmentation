#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip3 install -r runpod_requirements.txt
# Install dependencies
# pip3 install torch torchvision numpy matplotlib nibabel scikit-learn kagglehub
git config user.email ingridstoc@gmail.com
git config user.name ingridstoc
python3 runpod.py