import os
import argparse
import re
from PIL import Image, ImageEnhance

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, action='store', help='target directory')
parser.add_argument('--dryrun', action='store_true', default=False, help='dry-run flag')

args = parser.parse_args()
print('target directory=', args.directory)
if args.dryrun:
    print('dryrun!')

files = os.listdir(args.directory)
files = [file for file in files if not re.match('^\.', file)]
files = [file for file in files if ".c-" not in file]
for file in files:
    filepath = os.path.join(args.directory, file)
    filename, ext = os.path.splitext(filepath)

    img = Image.open(filepath)
    enhancer = ImageEnhance.Contrast(img)
    for p in range(10):
        output_filename = "%s.c-%d%s" % (filename, p + 1, ext)
        img_out = enhancer.enhance(1 - (p + 1) * 0.05)
        if not args.dryrun:
            img_out.save(output_filename, 'JPEG')
        print(output_filename)
