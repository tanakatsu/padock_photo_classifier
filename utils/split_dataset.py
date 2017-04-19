import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=float, action='store', default=0.1, help='test data ratio')
parser.add_argument('--uniform', action='store_true', default=False, help='uniform flag')
parser.add_argument('file', type=str, action='store', help='filelist')

args = parser.parse_args()

print('test data ratio=', args.test)

all_data = []
train_data = []
test_data = []


with open(args.file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        filepath, score = line.split(' ')
        all_data.append({'file': filepath, 'score': float(score)})

if args.test > 0:
    skip = int(1.0 / args.test)
    for i, data in enumerate(all_data):
        if i % skip == 0:
            test_data.append(data)
        else:
            train_data.append(data)

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
if args.uniform:
    print('train data size(uniformed)= %d (%d, %d, %d, %d)' % (min_num * 4, min_num, min_num, min_num, min_num))

with open('train.txt', 'w') as f:
    if args.uniform:
        cls_sprint = cls_sprint[:min_num]
        cls_mile = cls_mile[:min_num]
        cls_middle = cls_middle[:min_num]
        cls_long = cls_long[:min_num]

    for category in [cls_sprint, cls_mile, cls_middle, cls_long]:
        for data in category:
            f.write("%s %f\n" % (data['file'], data['score']))

cls_long = []
cls_middle = []
cls_mile = []
cls_sprint = []

for data in test_data:
    if data['score'] < 0.2:
        cls_sprint.append(data)
    elif data['score'] < 0.4:
        cls_mile.append(data)
    elif data['score'] < 0.6:
        cls_middle.append(data)
    else:
        cls_long.append(data)

print('test data size= %d (%d, %d, %d, %d)' % (len(test_data), len(cls_sprint), len(cls_mile), len(cls_middle), len(cls_long)))
min_num = min([len(cls_sprint), len(cls_mile), len(cls_middle), len(cls_long)])
if args.uniform:
    print('test data size(uniformed)= %d (%d, %d, %d, %d)' % (min_num * 4, min_num, min_num, min_num, min_num))

with open('test.txt', 'w') as f:
    if args.uniform:
        cls_sprint = cls_sprint[:min_num]
        cls_mile = cls_mile[:min_num]
        cls_middle = cls_middle[:min_num]
        cls_long = cls_long[:min_num]

    for category in [cls_sprint, cls_mile, cls_middle, cls_long]:
        for data in category:
            f.write("%s %f\n" % (data['file'], data['score']))
