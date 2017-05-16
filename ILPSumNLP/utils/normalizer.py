#!/usr/bin/python
# -*- coding: utf8 -*-
import string

import regex

from utils.utils import *


def is_punctuation(token):
    return all(c in string.punctuation for c in token)


def normalize_word_suffix(s, lang='en'):
    return regex.sub(r"\s+('m|'ll|'d|'s|n't)", '\g<1>', s, flags=regex.IGNORECASE) if lang is 'en' else s


def remove_punctuation(tokens):
    return [token for token in tokens if not is_punctuation(token)]


def normalize_special_chars(s):
    s = regex.sub(r'“|”', '"', s)
    s = regex.sub(r'`|‘|’', "'", s)
    s = regex.sub(r"''", '"', s)
    s = regex.sub(r'–', '-', s)
    s = regex.sub(r'…', '...', s)
    # Normalise extra white spaces
    s = regex.sub(r'[\s\xA0]+', ' ', s)
    return s.strip()


def remove_invalid_chars(s):
    s = normalize_special_chars(s)
    s = regex.sub(r'^\s*[!#%&\)*+,\-.\/:;=>?@\\\]^_|}~]\s*', '', s, flags=regex.MULTILINE)
    s = regex.sub(r'([!,.:;?])\s*[!#%&\)*+,\-\/:;=>?@\\\]^_|}~]\s*', '\g<1> ', s)
    s = regex.sub(r'([^\w %s])' % string.punctuation, '', s)
    s = regex.sub(r'\'\s*\'|"\s*"|<\s*>|{\s*}|\(\s*\)|\[\s*\]', '', s)
    return s.strip()


def parse_docs(raw_docs, stemmer, lang='en'):
    docs = []
    for raw_doc in raw_docs:
        doc = []

        parsed_sentences = parse(raw_doc['content'], lang)
        for pos, parsed_sentence in enumerate(parsed_sentences):
            sentence = {
                'doc_name': raw_doc['doc_name'],
                'pos': pos,
                'tokens': parsed_sentence['tokens'],
                'tags': ['PUNCT' if is_punctuation(token) else parsed_sentence['tags'][i] for i, token in
                         enumerate(parsed_sentence['tokens'])],
                'lemmas': parsed_sentence['lemmas'] if lang is 'en' else parsed_sentence['tokens'],
                'stemmers': [stemmer.stem(token) for token in
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

    return tokens
