import argparse
from urllib.request import urlretrieve
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('url', help='download url')
parser.add_argument('--file', action='store', type=str, default=None, help='filename to save')
args = parser.parse_args()

url = args.url
if args.file:
    filename = args.file
else:
    filename = os.path.basename(url)


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_file(url, filename):
    if not os.path.isfile(filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(filename)) as pbar:
            urlretrieve(url, filename, pbar.hook)

download_file(url, filename)
