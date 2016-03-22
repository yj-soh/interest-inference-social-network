import bs4
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import os
import re
import string
import csv

# utility functions for sorting files in directory
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

LINKEDIN_TRAINING_DIRECTORY = 'data/train/LinkedIn/'
LINKEDIN_TESTING_DIRECTORY = 'data/test/LinkedIn/'
LINKEDIN_TRAINING_FILE = 'data/generated/training_linkedin.txt'
LINKEDIN_TESTING_FILE = 'data/generated/testing_linkedin.txt'

stopwords = map(lambda s:str(s), stopwords.words('english'))

def parse_html(html):
    def parse_content(soup_content):
        words = []
        for html_content in soup_content:
            content = ' '.join([c for c in html_content.contents if not type(c) is bs4.element.Tag])
            tokens = [e.lower() for e in map(string.strip, re.split('(\W+)', content)) if len(e) > 0 and not re.match('\W',e)]
            for token in tokens:
                if not token in stopwords:

                    words.append(token)
        return words

    html_soup = BeautifulSoup(html, 'html.parser')

    words = []
    words.extend(parse_content(html_soup.findAll('p', {'class': 'title'})))
    words.extend(parse_content(html_soup.findAll('p', {'class': 'description'})))
    words.extend(parse_content(html_soup.findAll('a', {'class': 'endorse-item-name-text'})))

    # make bigrams
    bigrams = []
    for i, word in enumerate(words):
        if i - 1 >= 0:
            bigrams.append(words[i - 1] + '_' + words[i])

    words.extend(bigrams)

    return words

def parse(files):
    directory = files['in']
    output_file = files['out']
    directory_files = os.listdir(directory)
    directory_files.sort(key=alphanum_key)

    output_array = []
    for html_file in directory_files:
        with open(directory + html_file, 'r') as content_file:
            html = content_file.read()

        output_array.append(parse_html(html))

    f = open(output_file, 'wb')
    csv_writer = csv.writer(f, delimiter =' ')
    csv_writer.writerows(output_array)

if __name__ == '__main__':
    parse({
        'in': LINKEDIN_TRAINING_DIRECTORY,
        'out': LINKEDIN_TRAINING_FILE
    })
    parse({
        'in': LINKEDIN_TESTING_DIRECTORY,
        'out': LINKEDIN_TESTING_FILE
    })