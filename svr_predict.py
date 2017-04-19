from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import argparse

import image_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--test', default='test.txt', type=str, help='File name of test data')
parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
parser.add_argument('--model', '-m', default='model.pkl', help='model filename')

args = parser.parse_args()

test = image_dataset.ImageDataset(args.test, args.root, max_size=32)

X_test = []
y_test = []

print('Reading data...')
for data, score in test:
    X_test.append(data.reshape(-1))
    y_test.append(score[0])
print('done.')

# Load model
clf = joblib.load(args.model)

print('Predicting...')
y_pred = clf.predict(X_test)
print('done.')

print('mse=', mean_squared_error(y_test, y_pred))
