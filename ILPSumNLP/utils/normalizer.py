#!/usr/bin/python
# -*- coding: utf8 -*-
import string

import regex

from utils.utils import *


def is_punctuation(token):
    return all(c in string.punctuation for c in token)


def normalize_word_suffix(s, lang='en'):
    if lang is 'en':
        return regex.sub(r"\s+('m|'ll|'d|'s|n't)", '\g<1>', s, flags=regex.IGNORECASE)
    return s


def remove_punctuation(tokens):
    normalized_tokens = []
    for token in tokens:
        if not is_punctuation(token):
            normalized_tokens.append(token)
    return normalized_tokens


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


def get_tagged_words(sentence):
    pass


def parse_docs(raw_docs, stemmer, lang='en'):
    docs = []
    for raw_doc in raw_docs:
        doc = []

        parsed_sentences = parse(raw_doc, lang)
        for parsed_sentence in parsed_sentences:
            sentence = {
                'tokens': parsed_sentence['tokens'],
                'tags': ['PUNCT' if is_punctuation(token) else parsed_sentence['tags'][i] for i, token in
                         enumerate(parsed_sentence['tokens'])],
                'lemmas': parsed_sentence['lemmas'] if lang is 'en' else parsed_sentence['tokens'],
                'stemmer': [stemmer.stem(token) for token in
                            parsed_sentence['tokens']] if lang is 'en' else parsed_sentence['tokens']
            }
            doc.append(sentence)

        docs.append(doc)
    return docs


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
