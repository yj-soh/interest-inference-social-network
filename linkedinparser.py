import bs4
from bs4 import BeautifulSoup
import codecs
import nltk
from nltk.corpus import stopwords
import os
import re
import string
import csv

import textprocessor as parser

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

def extract_data(html_soup):
    words = {}

    words['name'] = html_soup.find('span', class_='full-name')

    def resolve(ele, f=lambda x: x):
        try:
            return map(lambda x: resolve(x, f), ele) if isinstance(ele, list) else f(ele)
        except:
            return [] if isinstance(f, list) else None

    def opt(f):
        try:
            return f()
        except:
            return []

    text = []

    text.extend([
        html_soup.find('p', class_='title'),
        html_soup.find('span', class_='locality'),
        html_soup.find('dd', class_='industry')
    ])

    for e in html_soup.find_all('div', class_='education'):
        text.extend([
            e.find('h4', class_='summary'),
            e.find('span', class_='degree'),
            resolve(e.find('p', class_='activities'), lambda m: m.a),
            e.find('p', class_='notes')
        ])
        text.extend(opt(lambda: e.find('span', class_='major').find_all('a')))

    text.extend(resolve(opt(lambda: html_soup.find_all('p', class_='following-name')),
                        lambda m: m.a.strong))

    text.extend(resolve(opt(lambda: html_soup.find('ul', class_='interests-listing').find_all('li')),
                        lambda m: m.a))

    for e in opt(lambda: html_soup.find_all('div', id=re.compile('experience-.*?-view'))):
        text.extend([
            e.header.h4,
            e.header.find(lambda t: t.name == 'h5' and not t.has_attr('class')),
            e.p
        ])

    text.extend(resolve(opt(lambda: html_soup.find_all('span', class_='skill-pill')),
                        lambda m: m.find_all('span')[1]))

    text.append(resolve(html_soup.find('div', class_='summary'), lambda m: m.p))

    text.extend(opt(lambda: html_soup.find('div', id='volunteering-opportunities').find_all('li')))

    for e in opt(lambda: html_soup.find_all('div', class_='experience')):
        text.extend([
            e.hgroup.h4,
            e.hgroup.h5,
            e.p
        ])

    text = filter(lambda x: x is not None, text)
    text = map(lambda x: x.get_text(), text)
    text = filter(lambda x: x is not None, text)
    text = parser.get_unigrams(parser.unigrams_to_str(text).lower())

    def should_remove(str):
        return parser.is_stopword(str) or \
        str.isdigit() or parser.is_punctuation(str) or \
        len(str) is 1 or parser.is_emoticon(str)

    text = (u for u in text if not should_remove(u))

    text = (word for word, emoji in map(parser.separate_emoji, text))
    text = map(parser.remove_at, text)
    text = map(parser.remove_hash, text)
    text = map(parser.trim_repeat_char, text)

    return text

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

    words = extract_data(html_soup)

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

        output_array.append(u' '.join(parse_html(html)))

    f = codecs.getwriter('utf8')(open(output_file, 'w'))
    for line in output_array:
        f.write(line + u'\n')
    f.close()

if __name__ == '__main__':
    parse({
        'in': LINKEDIN_TRAINING_DIRECTORY,
        'out': LINKEDIN_TRAINING_FILE
    })
    parse({
        'in': LINKEDIN_TESTING_DIRECTORY,
        'out': LINKEDIN_TESTING_FILE
    })