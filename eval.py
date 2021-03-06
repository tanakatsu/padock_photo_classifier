import os
import argparse
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import serializers
import chainer

import net
from PIL import Image

IMAGE_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument('filelist', type=str)
parser.add_argument('-m', '--model', type=str, required=True, help='model.npz')
parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
parser.add_argument('-n', type=int, default=None, help='sample numbers')
parser.add_argument('-s', '--step', type=int, default=1, help='step')
parser.add_argument('--mean', type=str, default=None, help='mean file (computed by compute_mean.py)')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

# load model
model = L.Classifier(net.CNN(), lossfun=F.mean_squared_error)
model.predictor.train = False

serializers.load_npz(args.model, model)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()

if args.mean:
    mean = np.load(args.mean)


def predict(x):
    y = model.predictor(x)
    pred = y.data
    score = pred[0]
    return score

input_data = []
line_cnt = 0
with open(args.filelist) as f:
    lines = f.readlines()
    for line in lines:
        if line_cnt % args.step == 0:
            line = line.strip()
            filepath, score = line.split(' ')
            img = Image.open(os.path.join(args.root, filepath))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            data = np.asarray(img).astype(np.float32).transpose(2, 0, 1)  # H, W, D -> D, H, W
            data /= 255
            if args.mean:
                data -= mean / 255.0
            input_data.append({"filename": filepath, "data": data, "score": score})
            if args.n:
                if len(input_data) >= args.n:
                    break
        line_cnt += 1

# pilImg = Image.fromarray(np.uint8(input_data[0]["data"].transpose(1, 2, 0) * 255))
# pilImg.save('test.jpg')

loss_history = []
d_history = []
for data in input_data:
    x = data["data"].reshape((1,) + data["data"].shape)
    if args.gpu >= 0:
        x = chainer.cuda.cupy.asarray(x, dtype=np.float32)
    score = predict(x)
    if args.gpu >= 0:
        score = chainer.cuda.to_cpu(score)
    data["score_n"] = float(data["score"]) * 9.0 + 1.0  # rescale to 1.0-10.0
    loss = F.mean_squared_error(Variable(np.array(score[0]).astype(np.float32)), Variable(np.array(float(data["score_n"])).astype(np.float32))).data
    d = abs(score[0] - float(data["score"]))
    loss_history.append(loss)
    d_history.append(d)
    modified_score = (score - 1.0) / 9.0  # rescale to 0.0-1.0
    print(data["filename"], data["score_n"], 'score=', score, data["score"], modified_score[0], 'loss=', "%.3f" % loss, 'd=', abs(score[0] - float(data["score"])))

loss_history = np.array(loss_history)
d_history = np.array(d_history)

print('average loss=', np.mean(loss_history), 'stdv=', np.std(loss_history))
print('average d=', np.mean(d_history), 'stdv=', np.std(d_history))
