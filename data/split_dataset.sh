#!/bin/bash

# python ../utils/split_dataset.py --even --test 160328 160502 160606 160627 161003 161024 161031 161121 --validation 16.+ --dataset dataset.txt
python ../utils/split_dataset.py --test 160328 160502 160606 160627 161003 161024 161031 161121 --validation 16.+ --dataset dataset.txt

cat train.txt | sed -n 'p;N;N;N;N' > train_mini.txt
