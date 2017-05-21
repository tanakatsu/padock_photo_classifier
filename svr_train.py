from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import argparse
import numpy as np
import csv

import image_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train', default='train.txt', type=str, help='File name of train data')
parser.add_argument('--test', default='validation.txt', type=str, help='File name of test data')
parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
parser.add_argument('--csv', default=None, help='output csv filename')
parser.add_argument('--output', '-o', default=None, help='output model filename')

args = parser.parse_args()

train = image_dataset.ImageDataset(args.train, args.root, max_size=32)
test = image_dataset.ImageDataset(args.test, args.root, max_size=32)

X_train = []
y_train = []
X_test = []
y_test = []

print('Reading data...')
for data, score in train:
    X_train.append(data.reshape(-1))
    y_train.append(score[0])
for data, score in test:
    X_test.append(data.reshape(-1))
    y_test.append(score[0])
print('done.')

X_train = np.array(X_train)
y_train = np.array(y_train)

print('Training...')
# clf = svm.SVR(C=5., gamma=0.001)
clf = Pipeline([('pca', PCA(svd_solver='randomized', n_components=1024, whiten=True, random_state=1999)),
                ('svm', svm.SVR(C=5., gamma=0.001))])
clf.fit(X_train, y_train)
print('done.')

# Save model
if args.output:
    joblib.dump(clf, args.output)

print('Predicting...')
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
print('done.')

print('train mse=', mean_squared_error(y_train, y_train_pred))
print('test mse=', mean_squared_error(y_test, y_test_pred))

# Output to csv
if args.csv:
    with open(args.csv, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for test, pred in zip(y_test, y_test_pred):
            data = []
            data.append(test)
            data.append(pred)
            data.append(abs(test - pred))
            writer.writerow(data)
