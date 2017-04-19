#!/bin/bash

awk '{print $1}' train.txt > train.mean.txt
python compute_mean.py --root data_32x32 train.mean.txt
rm train.mean.txt
