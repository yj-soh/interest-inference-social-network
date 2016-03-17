import json
import csv
import os

TWEET_DIR = 'data/tweets/'

def _read_json(filename):
    try:
        return json.load(open(filename))
    except ValueError:
        return ''

def _read_csv(filename, delimiter=','):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        reader.next() # skip header
        for row in reader:
            yield row

def read_tsv_map(tsvfile):
    tsv = {}
    for row in _read_csv(tsvfile, delimiter='\t'):
        tsv[row[0]] = row[1]
    return tsv

def read(jsondir):
    jsons = (_read_json(jsondir + '/' + f) for f in os.listdir(jsondir))
    for json in jsons:
      for tweet in json:
        yield tweet
