#! /usr/bin/env python

import argparse
import csv
from pprint import pprint
from time import time

import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion


def fetch_gitlog(data_path, stats=False):
    """Convert the CSV log into a datase suitable for scikit-learn."""
    description = 'Lageled git log history'

    with open(data_path, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"',
                               doublequote=True)
        next(csvreader)
        # Summary, Message, Number of Files, Total Lines, Added Lines, Deleted Lines, Label, SHA-1
        full_dataset = [(line[0].strip(), line[1].strip(),
                         int(line[2]), int(line[3]), int(line[4]),
                         int(line[5]), line[6], line[7]) for line in csvreader]

        data = np.array([d for d in full_dataset if d[-2]])

        # Encode targets into numbers
        target = [d[6] for d in data]
        le = preprocessing.LabelEncoder()
        le.fit(target)
        target_names = le.classes_
        target = le.transform(target)

        if stats:
            print 'Original dataset [%d]' % len(full_dataset)
            print 'Labeled dataset [%d]' % len(data)
            print 'Ratio [%2.2f%%]' % (100.0 * len(data) / len(full_dataset))

    return Bunch(filename=data_path,
                 data=data,
                 target_names=target_names,
                 target=target,
                 DESCR=description)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models with the same dataset.')
    parser.add_argument('-c', '--csv', help='csv file name')

    args = parser.parse_args()

    if not args.csv:
        parser.print_help()
        exit(1)

    data = fetch_gitlog(args.csv, stats=True)

    pipeline_summary = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ])

    pipeline_message = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ])

    l = data.data.shape[0]
    features_summary = pipeline_summary.fit_transform(data.data[:, [0]].reshape(l))
    features_message = pipeline_message.fit_transform(data.data[:, [1]].reshape(l))
    features_numeric = data.data[:, [2, 3, 4, 5]].astype(int)
    print 'SP1', features_summary.shape
    # print 'SP2', features_message.shape
    # print 'SP3', features_numeric.shape
    features = sparse.hstack((features_summary, features_message, features_numeric))

    print 'SPF', features.shape

    classifier = SGDClassifier()
    classifier.fit(features, data.target)

    print classifier.predict(features) - data.target
    print data.target

    # print features.getrow(10)
