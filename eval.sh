#!/bin/bash

if [ $# -ne 1 ]; then
  echo 'Error: trained model is not specified.'
  echo 'Usage: eval.sh trained_model'
  exit 1
fi

# python eval.py --model $1 --root data/32x32 --step 1 data/test.txt
python eval.py --model $1 --root data/32x32 --mean mean.npy --step 1 data/test.txt
