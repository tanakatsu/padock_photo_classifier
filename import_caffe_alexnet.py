import pickle
from chainer.links.caffe import CaffeFunction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', '-o', default='alexnet.pkl')
parser.add_argument('--input', '-i', default='bvlc_alexnet.caffemodel')
args = parser.parse_args()

loadpath = args.input
savepath = args.output

alexnet = CaffeFunction(loadpath)
pickle.dump(alexnet, open(savepath, 'wb'))
