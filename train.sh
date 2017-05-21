#!/bin/bash

# python train.py --train data/train.txt --test data/validation.txt --root data/32x32 -e 50 --gpu -1
python train.py --train data/train.txt --test data/validation.txt --root data/32x32 --mean mean.npy -e 50 --gpu -1
