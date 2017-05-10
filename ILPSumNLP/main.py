#!/usr/bin/python
# -*- coding: utf8 -*-
import kenlm

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import *
from utils.utils import *

LANGUAGE = 'en'

# Load language model
kenlm_model = kenlm.Model(full_path('../Resources/languagemodel/lm_giga_20k_vp_3gram.klm'))

# Load stopwords
STOPWORDS_EN = read_lines(full_path('../Resources/stopwords/en.txt'))
STOPWORDS_VI = read_lines(full_path('../Resources/stopwords/vi.txt'))

# Load Stemmer
SNOWBALL_STEMMER = SnowballStemmer('english')


def kenlm_score(sentence):
    num_words = len(sentence.split(' '))
    return kenlm_model.score(sentence) / (num_words + 1)


def eval_linguistic(method=None):
    pass


def clustering_sentences(docs, lang='en'):
    num_docs = len(docs)

    # Determine important document
    merged_docs = []
    raw_docs = []
    for doc in docs:
        raw_doc = []
        for sentence in doc:
            raw_doc.append(' '.join(remove_punctuation(sentence['stemmer'])))

        raw_doc = ' '.join(raw_doc)

        # Fix for English
        raw_doc = normalize_word_suffix(raw_doc, lang)

        merged_docs.append(raw_doc)
        raw_docs.append(raw_doc)

    raw_docs.append(' '.join(merged_docs))

    # Compute cosine similarities and get index of important document
    tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=STOPWORDS_EN if lang is 'en' else STOPWORDS_VI)
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_docs)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    imp_doc_idx = cosine_similarities[num_docs:, :num_docs].argmax()

    # Generate clusters
    clusters = []
    raw_sentences = []
    for sentence in docs[imp_doc_idx]:
        clusters.append({
            'tokens': sentence['tokens'],
            'tags': sentence['tags']
        })
        raw_sentences.append(normalize_word_suffix(' '.join(remove_punctuation(sentence['stemmer'])), lang))

    debug(raw_sentences)
    return clusters


def ordering_clusters(clusters):
    return clusters


def compress_clusters(clusters, lang='en'):
    clusters = []
    return clusters


def main():
    samples = read_json(full_path('../Temp/datasets/%s/packed.json' % LANGUAGE))
    for sample_name in samples.keys():
        debug('Cluster: %s' % sample_name)

        # Prepare docs in cluster
        raw_docs = samples[sample_name]['docs']
        docs = parse_docs(raw_docs, SNOWBALL_STEMMER, LANGUAGE)

        # Clustering sentences into clusters
        clusters = clustering_sentences(docs, LANGUAGE)
        clusters = ordering_clusters(clusters)
        clusters = compress_clusters(clusters, LANGUAGE)
        break


def prepare_datasets():
    preprocess_duc04('../Datasets/DUC04', '../Temp/datasets/en')
    preprocess_vimds('../Datasets/VietnameseMDS', '../Temp/datasets/vi')


if __name__ == '__main__':
    main()
