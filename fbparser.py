import codecs
import os
import re
import string

import textprocessor as parser

FILES = {
    'training': {
        'in': 'data/train/Facebook/',
        'out': 'data/generated/training_fb.txt'
    },
    'testing': {
        'in': 'data/test/Facebook/',
        'out': 'data/generated/testing_fb.txt'
    }
}

options = {
    'stopwords': False,
    'force_lowercase': True,
    'trim_repeat_char': True,
    'lemma': False,
    'replace_slang': False,
    'no_hash_hashtags': True
}

re_str_date = r'((January|February|March|April|May|June|July|August|September|October|November|December)\s\d+(,\s\d+)?\s)?at\s\d+:\d+(am|pm)'
re_str_actions = r'added \d+ new photos( to the album:|.)|uploaded a new video.|changed (his|her) profile picture.|likes a link on .+?.|likes an article.|shared a link.|shared .+?\'s photo.|updated (his|her) cover photo.'
re_str_social = r'\|By|View \d+ more comments?|View \d+ more [replies|reply]|\d+ Comments?|\d+ Likes?|\d+ [S|s]hares?|Share (\d+\speople|.+?) likes? this.'
re_str_static_terms = r'Automatically Translated  See Original|Comment|Edited|Like|Remove|Reply|See Translation|Share|Water Later|Write a comment... Press Enter to post.|\w+ emoticon'

re_fb_terms = re.compile(re_str_date + '|' + re_str_actions + '|' + re_str_social + '|' + re_str_static_terms, re.UNICODE)


def _parse_text(text):
    # remove fb terms
    text = re.sub(re_fb_terms, '', text)
    
    # parse terms
    unigrams = parser.get_unigrams(text)
    
    # remove stopwords, numbers, punctuation
    def should_remove(str):
        return parser.is_stopword(str) or \
        str.isdigit() or parser.is_punctuation(str) or \
        len(str) is 1 or parser.is_emoticon(str)
    
    unigrams = (u for u in unigrams if not should_remove(u))
    
    unigrams = (word for word, emoji in map(parser.separate_emoji, unigrams))
    unigrams = map(parser.remove_at, unigrams)
    unigrams = map(parser.remove_hash, unigrams)
    unigrams = map(parser.trim_repeat_char, unigrams)
    
    text = parser.unigrams_to_str(unigrams)
    
    # force lowercase
    text = text.lower()
    
    return text

def _parse_fb(fb_file):
    with open(fb_file, 'r') as file:
        text = file.read()
        text = _parse_text(text)
        return text

def parse_all_files(new_options=options):
    dirs = [FILES['training'], FILES['testing']]
    
    options = new_options
    
    for type in dirs:
        fb_dir = type['in']
    
        # toss everything into memory; should be fine due to data's size
        
        files = os.listdir(fb_dir)
        files = [fn[1:] for fn in files]
        files = sorted(sorted(files), key=lambda s: len(s))
        files = ['U' + fn for fn in files]
        
        f = codecs.getwriter('utf8')(open(type['out'], 'w'))
        for file in files:
            line = _parse_fb(fb_dir + file)
            line = u' '.join(line.splitlines()) # rejoin item if it contains newline(s)
            
            f.write(line + u'\n')
        f.close()

if __name__ == '__main__':
        parse_all_files()
