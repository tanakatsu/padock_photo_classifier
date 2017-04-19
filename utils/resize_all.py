import argparse
import os
import re
import resizer

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, action='store', default='.')
parser.add_argument('--width', type=int, action='store', default=32)
parser.add_argument('--height', type=int, action='store', default=32)
parser.add_argument('input_dir', type=str, action='store')

args = parser.parse_args()

if args.input_dir == args.output_dir:
    print('Error: output directory is same as input directory.')
    exit()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print("resize target=(%d, %d)" % (args.width, args.height))
files = os.listdir(args.input_dir)
files = [file for file in files if not re.match('^\.', file)]
for i, file in enumerate(files):
    inputname = os.path.join(args.input_dir, file)
    outputname = os.path.join(args.output_dir, file)
    print('%d: input=%s, output=%s' % (i, inputname, outputname))
    resizer.resize(inputname, outputname, args.width, args.height)
