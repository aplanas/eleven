#! /usr/bin/env python

import argparse
import csv
from pprint import pprint
import re
from time import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import Binarizer, MinMaxScaler
from sklearn.svm import LinearSVC


def fetch_gitlog(data_path, collapse=None, full_data=False, stats=False):
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

        if not full_data:
            data = np.array([d for d in full_dataset if d[6]])
        else:
            data = np.array(full_dataset)

        collapse = {} if not collapse else collapse
        for key, value in collapse.items():
            data[data[:, 6] == key, 6] = value

        # Encode targets into numbers
        target = [d[6] for d in data]
        le = LabelEncoder()
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


class SliceFeature(BaseEstimator):
    """Estimator to slice a feature."""

    def __init__(self, slc, astype=None, flatten=False):
        """Build an instance using a slice object.

        >>> X = np.array([[1, 2, 3], [10, 20, 30]])
        >>> X
        array([[ 1,  2,  3],
               [10, 20, 30]])

        >>> slc = SliceFeature(slice(0, 1))
        >>> slc.transform(X)
        array([ 1, 10])

        >>> slc = SliceFeature(slice(0, 2))
        >>> slc.transform(X)
        array([[ 1,  2],
               [10, 20]])

        """
        self.slc = slc
        self.astype = astype
        self.flatten = flatten

    def fit(self, X, y=None):
        """Does nothing: this transformer is stateless."""
        return self

    def transform(self, X, y=None):
        if self.slc.step:
            index = range(self.slc.start, self.slc.stop, self.slc.step)
        else:
            index = range(self.slc.start, self.slc.stop)

        result = X[:, index]

        if self.astype:
            result = result.astype(self.astype)

        if self.flatten:
            result = result.reshape(X.shape[0])

        return result


class RegexSpotter(BaseEstimator):
    def __init__(self, regexp):
        # store the actual argument, so BaseEstimator.get_params() will do it's magic.
        self.regexp = regexp
        self.pattern = re.compile(regexp)

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        matches = np.fromiter((self.pattern.search(x) for x in X), dtype=bool)
        return matches[:, np.newaxis]


def regex_pipeline(column, regex):
    pipeline = Pipeline([
        ('slice', SliceFeature(slice(column, column + 1), flatten=True)),
        ('sha_spotter', RegexSpotter(regex))
    ])
    return pipeline


class Densifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return X.toarray()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models with the same dataset.')
    parser.add_argument('-c', '--csv', help='csv file name')
    parser.add_argument('-l', '--label', action='store_true', help='label missing data')
    parser.add_argument('-d', '--debug', help='turn on debugging: only one job', action='store_true')

    args = parser.parse_args()

    if not args.csv:
        parser.print_help()
        exit(1)

    if args.debug:
        n_jobs = 1
    else:
        n_jobs = -1

    pipeline_summary = Pipeline([
        ('slice', SliceFeature(slice(0, 1), flatten=True)),
        ('vect', CountVectorizer(stop_words='english')),
        # ('binary', Binarizer()),
        ('tfidf', TfidfTransformer()),
        # ('scaler', StandardScaler(with_mean=False)),
    ])

    pipeline_message = Pipeline([
        ('slice', SliceFeature(slice(1, 2), flatten=True)),
        ('vect', CountVectorizer(stop_words='english')),
        # ('binary', Binarizer()),
        ('tfidf', TfidfTransformer()),
        # ('scaler', StandardScaler(with_mean=False)),
    ])

    pipeline_numeric = Pipeline([
        ('slice', SliceFeature(slice(2, 6), astype=int)),
    ])

    main_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('summary', pipeline_summary),
            ('message', pipeline_message),
            ('numeric', pipeline_numeric),
            ('contains_sha', regex_pipeline(1, r'[0-9a-eA-E]{6,}')),
            # ('contains_http', regex_pipeline(1, r'https?://')),
            # ('contains_bugzilla', regex_pipeline(1, r'bugzilla\.kernel\.org')),
            # ('contains_lkml', regex_pipeline(1, r'lkml\.kernel\.org')),
        ])),
        # ('densifier', Densifier()),
        # ('scaler', StandardScaler(with_mean=False)),
        # ('scaler', StandardScaler()),
        ('clf', LinearSVC()),
        # ('clf', LogisticRegression()),
    ])

    parameters = {
        'features__summary__vect__max_df': (0.25, 0.5),
        # 'features__summary__vect__max_df': (0.5, 0.75, 1.0),
        'features__summary__vect__max_features': (None, 10, 100, 1000),
        # 'features__summary__vect__max_features': (None, 5000, 10000, 50000),
        # 'features__summary__vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'features__summary__tfidf__use_idf': (True, False),
        # 'features__summary__tfidf__norm': ('l1', 'l2'),
        'features__message__vect__max_df': (0.5, 0.75, 1.0),
        'features__message__vect__max_features': (None, 100, 1000, 5000),
        # 'features__message__vect__max_features': (None, 5000, 10000, 50000),
        # 'features__message__vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'features__message__tfidf__use_idf': (True, False),
        # 'features__message__tfidf__norm': ('l1', 'l2'),
        # 'clf__C': (0.0001, 0.001, 0.01, 0.1, 1.0),
        # 'clf__loss': ('l1', 'l2'),
        # 'clf__penalty': ('l1', 'l2'),
        # 'clf__dual': (True, False),
        # 'clf__tol': (1e-4),
        # 'clf__multi_class': ('ovr', 'crammer_singer'),
        # 'clf__fit_intercept': (True, False),
        # 'clf__intercept_scaling': (0.0001, 0.001, 0.01, 0.1, 1.0),
    }
    grid_search = GridSearchCV(main_pipeline, parameters, n_jobs=n_jobs, verbose=1)

    print 'Performing grid search...'
    print 'pipeline:', [name for name, _ in main_pipeline.steps]
    print 'parameters:'
    pprint(parameters)

    collapse_map = {
        'Fix (Minor)': 'Fix',
        'Fix (Major)': 'Fix',
        'Regression (Minor)': 'Regression',
        'Regression (Major)': 'Regression',
        # 'Regression (Minor)': 'Fix',
        # 'Regression (Major)': 'Fix',
        # 'Refactoring (Minor)': 'Refactoring',
        # 'Refactoring (Major)': 'Refactoring',
        'Refactoring (Minor)': 'Feature',
        'Refactoring (Major)': 'Feature',
        'Feature (Minor)': 'Feature',
        'Feature (Major)': 'Feature',
        'Documentation (Minor)': 'Documentation',
        'Documentation (Major)': 'Documentation',
        # 'Documentation (Minor)': 'Feature',
        # 'Documentation (Major)': 'Feature',
    }
    data = fetch_gitlog(args.csv, collapse=collapse_map, stats=True)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

    t0 = time()

    grid_search.fit(X_train, y_train)
    print 'done in %0.3fs' % (time() - t0)
    print

    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    print

    # Print the confusion matrix
    estimator = grid_search.best_estimator_
    y_pred = estimator.predict(X_test)

    print 'Confusion matrix for', data.target_names
    print confusion_matrix(y_test, y_pred)

    if args.label:
        # Get the full data and add labels
        full_data = fetch_gitlog(args.csv, full_data=True)
        unknown = [i for i, l in enumerate(full_data.target_names) if not l][0]
        y_pred = estimator.predict(full_data.data)
        for x, y, y_p in zip(full_data.data, full_data.target, y_pred):
            if not y:
                print data.target_names[y_p]
            else:
                print full_data.target_names[y]
