#!/usr/bin/python
# -*- coding: utf8 -*-
import string
from itertools import repeat

import regex

from utils.utils import *


# OK
def is_punctuation(token):
    return all(c in string.punctuation for c in token)


# OK
def normalize_word_suffix(s, lang='en'):
    return regex.sub(r"\s+('m|'d|'s|'ll|'re|'ve|n't)", '\g<1>', s, flags=regex.IGNORECASE) if lang == 'en' else s


# OK
def remove_punctuation(tokens):
    return list(filter(lambda x: not is_punctuation(x), tokens))


# OK
def normalize_special_chars(s):
    # s = regex.sub(r"“|”|``|''", '"', s)
    # s = regex.sub(r'`|‘|’', "'", s)
    # s = regex.sub(r'–', '-', s)
    # s = regex.sub(r'…', '...', s)
    # Normalise extra white spaces
    s = regex.sub(r'\s+', ' ', s)
    return s.strip()


# OK
def get_num_words(s):
    s = normalize_special_chars(s)
    return len(remove_punctuation(regex.split('\s+', s)))


# OK
def remove_underscore(s):
    s = regex.sub(r'([^ ])_([^ ])', '\g<1> \g<2>', s)
    s = regex.sub(r'_([^ ])', '\g<1>', s)
    s = regex.sub(r'([^ ])_', '\g<1>', s)
    # Normalise extra white spaces
    s = regex.sub(r'\s+', ' ', s)
    return s.strip()


# OK
def normalize_punctuation(s):
    s = regex.sub(r'\s*"\s+([^"]+)\s+"\s*', ' "\g<1>" ', s)
    s = regex.sub(r'\s+([\)\]}])\s*', '\g<1> ', s)
    s = regex.sub(r'\s*([\(\[{])\s+', ' \g<1>', s)
    s = regex.sub(r'\s+([!,.:;?])\s*', '\g<1> ', s)
    s = regex.sub(r'([!,.:;?])\s*(?=[!,.:;?\)\]}"])', '\g<1>', s)
    # Normalise extra white spaces
    s = regex.sub(r'\s+', ' ', s)
    return s.strip()


# OK
def parse_doc(raw_doc, lang='en'):
    sentences = []
    parsed_sentences = parse(raw_doc['content'], lang)
    for pos, parsed_sentence in enumerate(parsed_sentences):
        sentences.append({
            'name': raw_doc['name'],
            'pos': pos,
            'sentence': ' '.join(parsed_sentence['tokens']),
            'tokens': parsed_sentence['tokens'],
            'tags': ['PUNCT' if is_punctuation(token) else parsed_sentence['tags'][i] for i, token in
                     enumerate(parsed_sentence['tokens'])]
        })
    return sentences


# OK
def parse_docs(raw_docs, lang='en'):
    return pool_executor(fn=parse_doc, args=[raw_docs, repeat(lang)], executor_mode=1 if lang == 'en' else 0)
