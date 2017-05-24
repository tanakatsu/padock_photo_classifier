import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('logfiles', type=str, nargs='+', help='log files')
parser.add_argument('-x', type=str, default='epoch')
# parser.add_argument('-y', type=str, default='validation/main/accuracy')
parser.add_argument('-y', type=str, default='validation/main/loss')
parser.add_argument('-y2', type=str, default='main/loss')
parser.add_argument('--xlabel', type=str, default='epoch')
parser.add_argument('--ylabel', type=str, default='loss')
parser.add_argument('-o', '--output', type=str, default=None)
args = parser.parse_args()

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)

for file in args.logfiles:
    x = []
    y = []
    y2 = []
    with open(file, 'r') as f:
        log = json.load(f)
        for data in log:
            x.append(data[args.x])
            y.append(data[args.y])
            y2.append(data[args.y2])
    plt.plot(x, y, label='validation loss')
    plt.plot(x, y2, label='training loss')
#plt.legend(loc='upper left')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8))

if args.output:
    plt.savefig(args.output)

plt.show()
