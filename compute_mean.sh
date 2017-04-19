#!/bin/bash

awk '{print $1}' data/train.txt > data/train.mean.txt
python compute_mean.py --root data/32x32 data/train.mean.txt
rm data/train.mean.txt
