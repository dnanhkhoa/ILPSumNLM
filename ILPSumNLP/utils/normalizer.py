#!/usr/bin/python
# -*- coding: utf8 -*-
import string

import regex

from utils.utils import *


def remove_punctuation(sentence):
    return sentence


def normalize_special_chars(s):
    s = regex.sub(r'“|”', '"', s)
    s = regex.sub(r'`|‘|’', "'", s)
    s = regex.sub(r"''", '"', s)
    s = regex.sub(r'–', '-', s)
    s = regex.sub(r'…', '...', s)
    s = regex.sub(r'[\s\xA0]+', ' ', s)
    return s.strip()


def remove_invalid_chars(s):
    s = normalize_special_chars(s)
    s = regex.sub(r'^\s*[!#%&\)*+,\-.\/:;=>?@\\\]^_|}~]\s*', '', s, flags=regex.MULTILINE)
    s = regex.sub(r'([!,.:;?])\s*[!#%&\)*+,\-\/:;=>?@\\\]^_|}~]\s*', '\g<1> ', s)
    s = regex.sub(r'([^\w %s])' % string.punctuation, '', s)
    s = regex.sub(r'\'\s*\'|"\s*"|<\s*>|{\s*}|\(\s*\)|\[\s*\]', '', s)
    return s.strip()


def normalize_sentence(sentence, lang='en'):
    return sentence


def normalize_dataset(doc, lang='en'):
    doc = remove_invalid_chars(doc)

    tokens = []
    sentences = parse(doc, lang)
    for sentence in sentences:
        for token in sentence['tokens']:
            if token is '_':  # Remove "_"
                continue
            tokens.append(regex.sub('_+', ' ', token.lower()).strip())

    return ' '.join(tokens)
