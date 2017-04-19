import os
import argparse
import re
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, action='store', help='target directory')
parser.add_argument('--dryrun', action='store_true', default=False, help='dry-run flag')

args = parser.parse_args()
print('target directory=', args.directory)
if args.dryrun:
    print('dryrun!')

files = os.listdir(args.directory)
files = [file for file in files if not re.match('^\.', file)]
files = [file for file in files if ".flip." not in file]
for file in files:
    filepath = os.path.join(args.directory, file)
    filename, ext = os.path.splitext(filepath)
    output_filename = "%s.flip%s" % (filename, ext)
    img = Image.open(filepath)
    mirror_img = ImageOps.mirror(img)
    if not args.dryrun:
        mirror_img.save(output_filename, 'JPEG')
    print(output_filename)
