#! /usr/bin/env python

import argparse
import csv


def fetch_gitlog(data_path, stats=False):
    """Convert the CSV log into a datase suitable for scikit-learn."""
    with open(data_path, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"',
                               doublequote=True)
        next(csvreader)
        # Summary, Message, Number of Files, Total Lines, Added Lines, Deleted Lines, Label, SHA-1
        full_dataset = [(line[0].strip(), line[1].strip(),
                         int(line[2]), int(line[3]), int(line[4]),
                         int(line[5]), line[6], line[7]) for line in csvreader]

        dataset = [d for d in full_dataset if d[-2]]

        if stats:
            print 'Original dataset [%d]' % len(full_dataset)
            print 'Labeled dataset [%d]' % len(dataset)
            print 'Ratio [%2.2f%%]' % (100.0 * len(dataset) / len(full_dataset))

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models with the same dataset.')
    parser.add_argument('-c', '--csv', help='csv file name')

    args = parser.parse_args()

    if not args.csv:
        parser.print_help()
        exit(1)

    dataset = fetch_gitlog(args.csv, stats=True)
