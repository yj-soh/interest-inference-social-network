import HTMLParser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import string

import tweetreader as reader

ADVERB_TAGS = ['RB', 'RBR', 'RBS', 'WRB']
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'WP']
ADJECTIVE_TAGS = ['JJ', 'JJR', 'JJS']
VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

html_parser = HTMLParser.HTMLParser()
pos_tagger = nltk.tag.perceptron.PerceptronTagger()
punctuation = unicode(string.punctuation)
lemmatizer = WordNetLemmatizer()
stopwords = map(lambda s:str(s), stopwords.words('english'))
slang = {k:v.lower() for k, v in reader.read_tsv_map('resources/noslang.csv').items()}

re_str_emoji = u'\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|[\u2600-\u26FF\u2700-\u27BF]+'

re_str_emoticon = r'''
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )'''

re_str_words_meta = r'''
    (?:@[\w]+)                     # Usernames
    |
    (?:\#+[\w]+[\w\'\-]*[\w]+)     # Hashtags
    |
    (?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,}) # URLs
    '''
re_str_words = r'''
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w]+)                      # Words without apostrophes or dashes.
    '''
re_str_words_etc = r'''
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    '''

re_emoji = re.compile(re_str_emoji, re.UNICODE)
re_emoticon = re.compile(re_str_emoticon, re.VERBOSE | re.I | re.UNICODE)
re_words = re.compile(re_str_emoticon + '|' + re_str_words_meta + '|' + re_str_words + '|' + re_str_words_etc, \
                      re.VERBOSE | re.I | re.UNICODE)
re_repeat_char = re.compile(r'(.)\1+')
re_numbers = re.compile(r'\d+')
re_punctuation = re.compile('[%s]' % punctuation)

def get_unigrams(text_str):
    return re_words.findall(text_str)

def pos_tag(text_ls):
    return nltk.tag._pos_tag(text_ls, None, pos_tagger)

def replace_slang(text_ls):
    words = []
    for word in text_ls:
        if word.lower() in slang:
            words.extend(slang[word.lower()].split())
        else:
            words.append(word)
    return words

def is_emoticon(unigram_str):
    return re_emoticon.search(unigram_str)

def remove_hash(hashtag_str):
    if hashtag_str.startswith('#'):
        return hashtag_str[1:]
    return hashtag_str

def remove_at(at_str):
    if at_str.startswith('@'):
        return at_str[1:]
    return at_str

def unigrams_to_str(text_ls):
    return u' '.join(u' '.join(u' '.join(text_ls).splitlines()).split())

def is_stopword(unigram_str):
    return unigram_str in stopwords

def trim_repeat_char(unigram_str):
    return re_repeat_char.sub(r'\1\1', unigram_str)

def lemmatize(unigram_str, tag_str):
    try:
        return str(lemmatizer.lemmatize(unigram_str, _get_tag_type(tag_str)))
    except UnicodeDecodeError:
        return unigram_str

def separate_emoji(text_str):
    emoji = re_emoji.findall(text_str)
    text = re_emoji.sub('', text_str)
    
    return text, emoji

def normalize_markup(text_str):
    return unicode(html_parser.unescape(text_str))

def is_punctuation(text_str):
    return all(c in punctuation for c in text_str)

def remove_punctuation(text_str):
    return re.sub(re_punctuation, ' ', text_str)

def remove_numbers(text_str):
    return re.sub(re_numbers, '', text_str)
