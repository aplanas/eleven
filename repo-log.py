#! /usr/bin/env python

import argparse
import csv

import git


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract git history information.')
    parser.add_argument('-f', '--from', dest='from_', help='from revno')
    parser.add_argument('-t', '--to', help='to revno')
    parser.add_argument('-l', '--limit', help='max number of commits')
    parser.add_argument('-p', '--project', help='project directory')
    parser.add_argument('-c', '--csv', help='csv file name')

    args = parser.parse_args()

    if not args.csv or not args.project:
        parser.print_help()
        exit(1)

    with open(args.csv, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='|')
        repo = git.Repo(args.project)
        if args.limit:
            iter_ = repo.iter_commits(args.from_, max_count=args.limit)
        else:
            iter_ = repo.iter_commits(args.from_)
        for commit in iter_:
            if commit.hexsha == args.to:
                break
            summary = commit.summary.encode('utf-8')
            message = commit.message.encode('utf-8')
            stats = commit.stats.total
            csvwriter.writerow((summary, message, stats['files'],
                                stats['lines'], stats['insertions'],
                                stats['deletions']))
