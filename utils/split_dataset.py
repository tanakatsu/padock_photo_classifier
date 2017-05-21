import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--validation', type=str, action='store', nargs='+', required=True, help='validation data')
parser.add_argument('--test', type=str, action='store', nargs='+', required=True, help='test data')
parser.add_argument('--even', action='store_true', default=False, help='even flag')
parser.add_argument('--dataset', type=str, action='store', default='dataset.txt', help='filelist')

args = parser.parse_args()

train_data = []
validation_data = []
test_data = []

with open(args.dataset, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filepath, score = line.split(' ')

        data_type = None
        for param in args.test:
            if re.match("^" + param, filepath):
                test_data.append({'file': filepath, 'score': float(score)})
                data_type = "test"
                break

        if not data_type:
            for param in args.validation:
                if re.match("^" + param, filepath):
                    validation_data.append({'file': filepath, 'score': float(score)})
                    data_type = "validation"
                    break

        if not data_type:
            train_data.append({'file': filepath, 'score': float(score)})
            data_type = "train"

#
# train data
#
cls_long = []
cls_middle = []
cls_mile = []
cls_sprint = []

for data in train_data:
    if data['score'] < 0.2:
        cls_sprint.append(data)
    elif data['score'] < 0.4:
        cls_mile.append(data)
    elif data['score'] < 0.6:
        cls_middle.append(data)
    else:
        cls_long.append(data)

print('train data size= %d (%d, %d, %d, %d)' % (len(train_data), len(cls_sprint), len(cls_mile), len(cls_middle), len(cls_long)))
min_num = min([len(cls_sprint), len(cls_mile), len(cls_middle), len(cls_long)])
if args.even:
    print('train data size(even)= %d (%d, %d, %d, %d)' % (min_num * 4, min_num, min_num, min_num, min_num))

with open('train.txt', 'w') as f:
    if args.even:
        cls_sprint = cls_sprint[:min_num]
        cls_mile = cls_mile[:min_num]
        cls_middle = cls_middle[:min_num]
        # cls_mile = cls_mile[:min_num // 4]
        # cls_middle = cls_middle[:min_num // 4]
        cls_long = cls_long[:min_num]

    for category in [cls_sprint, cls_mile, cls_middle, cls_long]:
        for data in category:
            f.write("%s %f\n" % (data['file'], data['score']))

#
# validation data
#
cls_long = []
cls_middle = []
cls_mile = []
cls_sprint = []

for data in validation_data:
    if data['score'] < 0.2:
        cls_sprint.append(data)
    elif data['score'] < 0.4:
        cls_mile.append(data)
    elif data['score'] < 0.6:
        cls_middle.append(data)
    else:
        cls_long.append(data)

print('validation data size= %d (%d, %d, %d, %d)' % (len(validation_data), len(cls_sprint), len(cls_mile), len(cls_middle), len(cls_long)))
min_num = min([len(cls_sprint), len(cls_mile), len(cls_middle), len(cls_long)])
if args.even:
    print('validation data size(even)= %d (%d, %d, %d, %d)' % (min_num * 4, min_num, min_num, min_num, min_num))

with open('validation.txt', 'w') as f:
    if args.even:
        cls_sprint = cls_sprint[:min_num]
        cls_mile = cls_mile[:min_num]
        cls_middle = cls_middle[:min_num]
        cls_long = cls_long[:min_num]

    for category in [cls_sprint, cls_mile, cls_middle, cls_long]:
        for data in category:
            f.write("%s %f\n" % (data['file'], data['score']))

#
# test data
#
with open('test.txt', 'w') as f:
    for data in test_data:
        f.write("%s %f\n" % (data['file'], data['score']))
