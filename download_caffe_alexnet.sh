#! /bin/bash

if [ ! -f bvlc_alexnet.caffemodel ]; then
  echo 'downloading...'
  wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
fi
echo 'downloaded'
