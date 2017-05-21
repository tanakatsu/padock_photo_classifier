#!/bin/bash

# python eval.py --model $1 --root data/32x32 --step 1 data/test.txt
python eval.py --model $1 --root data/32x32 --mean mean.npy --step 1 data/test.txt
