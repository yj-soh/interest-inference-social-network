import codecs
import os
import pickle

import tweetreader as reader
import textprocessor as parser

FILES = {
    'training': {
        'in': 'data/train/Twitter/',
        'out': 'data/generated/training_tweets.txt'
    },
    'testing': {
        'in': 'data/test/Twitter/',
        'out': 'data/generated/testing_tweets.txt'
    }
}

options = {
    'stopwords': True,
    'force_lowercase': True,
    'trim_repeat_char': True,
    'lemma': False,
    'replace_slang': False,
    'no_hash_hashtags': True,
    'no_at': True,
    'no_numbers': True,
    'no_one_char': True
}

def _get_unigrams(text):
    words = parser.get_unigrams(text)
    
    if options['replace_slang']:
        words = parser.replace_slang(words)
    
    tagged_words = parser.pos_tag(words)
    
    return tagged_words

def _process_word(word, tag):
    # if is emoticon
    if parser.is_emoticon(word):
        return ''
    
    ### lower-case operations below ###
    if options['force_lowercase']:
        word = word.lower()
    
    if options['no_hash_hashtags']:
        word = parser.remove_hash(word)
    
    if options['no_at']:
        word = parser.remove_at(word)
    
    if options['no_numbers'] and word.isdigit():
        return ''
    
    if options['no_one_char'] and len(word) is 1:
        return ''
    
    # if is stopword
    if options['stopwords'] and parser.is_stopword(word):
        return ''
    
    if options['trim_repeat_char']:
        word = parser.trim_repeat_char(word)
    
    if options['lemma']:
        word = parser.lemmatize(word, tag)
    
    return word

def _parse_text(tweet):
    # extract emoji
    tweet, emoji = parser.separate_emoji(tweet)
    
    # markup normalization
    tweet = parser.normalize_markup(tweet)
    
    # split into unigrams
    tagged_words = _get_unigrams(tweet)
    
    # process each unigram
    rtweet = []
    
    for tagged_word in tagged_words:
        word, tag = tagged_word
        word = _process_word(word, tag)
        if word:
            rtweet.append((word, tag))
    
    # unpack for words-only processing
    words = [w[0] for w in rtweet]
    tags = [w[1] for w in rtweet]
    
    # after-splitting operations
    # rtweet = remove punctuation?
    
    # repack into tuples
    rtweet = zip(words, tags)
    
    return [w[0] for w in rtweet], rtweet

def _parse_tweets(tweets_dir, f):
    for json in reader.read(tweets_dir):
        tweet = {}
        
        # text
        tweet['text'] = json['text']
        tweet['unigrams'], tweet['tagged_unigrams'] = _parse_text(json['text'])
        line = u' '.join(u for u in tweet['unigrams'] if not parser.is_punctuation(u))
        f(line)

def parse(type, new_options=options):
    options = new_options
    
    all_tweets_dir = type['in']

    # toss everything into memory; should be fine due to data's size
    tweets = []
    
    dirs = os.listdir(all_tweets_dir)
    dirs = [d[1:] for d in dirs]
    dirs = sorted(sorted(dirs), key=lambda s: len(s))
    dirs = ['U' + d for d in dirs]
    
    f = codecs.getwriter('utf8')(open(type['out'], 'w'))
    for dir in dirs:
        tweets = []
        
        def collect(tweet):
            tweets.append(tweet)
        _parse_tweets(all_tweets_dir + dir, collect)
        
        line = parser.unigrams_to_str(tweets)
        
        f.write(line + u'\n')
        
    f.close()

if __name__ == '__main__':
    parse(FILES['training'])
    parse(FILES['testing'])
