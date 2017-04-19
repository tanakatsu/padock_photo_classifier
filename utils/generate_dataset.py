import pickle
import argparse
import os
import re


def load_file(filename):
    # http://qiita.com/Kodaira_/items/91207a7e092f491fca43
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('photo_file', help='padock photo urls pkl')
parser.add_argument('score_file', help='distance aptitude score pkl')
parser.add_argument('data_directory', help='image data directory')
parser.add_argument('--output', '-o', action='store', type=str, default='dataset.txt')
args = parser.parse_args()

padock_photo_data = load_file(args.photo_file)
score_data = load_file(args.score_file)

filenames = os.listdir(args.data_directory)
filenames = [file for file in filenames if not re.match('^\.', file)]

data_dict = []
for data in padock_photo_data:
    url = data['padock_photo_url']
    params = url.replace('http://', '').split('/')
    filename = '%s_%s' % (params[2], params[4])
    filename = filename.replace('.jpg', '')
    name = data['name']
    if name in score_data:
        score = score_data[name]
    else:
        score = None

    if score:
        data_dict.append({'name': name, 'score': score, 'filename': filename})

sorted_data_dict = sorted(data_dict, key=lambda x: -x['score'])
# print(sorted_data_dict)

if args.output:
    with open(args.output, 'w') as f:
        for filename in filenames:
            print(filename)
            filename_prefix = filename.split('.')[0]

            try:
                data_item = next((item for item in sorted_data_dict if item["filename"] == filename_prefix))
                print(data_item)
                print("%s %f\n" % (filename, data_item['score']))
                f.write("%s %f\n" % (filename, data_item['score']))
            except Exception:
                pass
else:
    for filename in filenames:
        print(filename)
        filename_prefix = filename.split('.')[0]

        try:
            data_item = next((item for item in sorted_data_dict if item["filename"] == filename_prefix))
            print(data_item)
            print("%s %f\n" % (filename, data_item['score']))
        except Exception:
            pass
