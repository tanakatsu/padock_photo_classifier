#!/bin/bash

python train.py --train data/train.txt --test data/test.txt --root data/32x32 --mean mean.npy -e 50 --gpu -1
# python train.py --train data/train_mini.txt --test data/test_mini.txt --root data/32x32 --mean mean.npy -e 50 --gpu -1
