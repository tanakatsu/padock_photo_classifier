#!/bin/bash

# python eval.py --model result/model_iter_83263 --root data/32x32 --step 10 data/test_mini.txt
python eval.py --model result/model_iter_83263 --root data/32x32 --mean mean.npy --step 10 data/test_mini.txt
