#!/bin/bash

echo 'Resizing images...'
resize_all.sh

echo 'Creating fliped images...'
apply_flipping.sh

echo 'Creating contrast jiggling images...'
apply_contrast_jiggling.sh

echo 'Generating dataset.txt...'
generate_dataset.sh

echo 'Splitting dataset.txt into train.txt, test.txt...'
split_dataset.sh

echo 'all done.'
