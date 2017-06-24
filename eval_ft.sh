#!/bin/bash

if [ $# -ne 1 ]; then
  echo 'Error: trained model is not specified.'
  echo 'Usage: eval.sh trained_model'
  exit 1
fi

python eval_ft.py --model $1 --root data/128x128 --mean mean.128px.npy --step 1 data/test.txt
