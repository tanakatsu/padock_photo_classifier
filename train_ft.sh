#!/bin/bash

python train_ft.py --train data/train.128px.txt --test data/validation.txt --root data/128x128 --mean mean.128px.npy -e 10 --gpu -1
