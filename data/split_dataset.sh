#!/bin/bash

python ../utils/split_dataset.py --uniform --test 0.1 dataset.txt

cat train.txt | sed -n 'p;N;N;N;N' > train_mini.txt
cat test.txt | sed -n 'p;N;N;N;N' > test_mini.txt
