#!/usr/bin/python
# -*- coding: utf8 -*-
import kenlm
import math

import networkx as nx
from nltk.stem import SnowballStemmer
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core import WordGraph
from utils.preprocessing import *
from utils.utils import *

# Configs

PREPARE_DATASET = False

LANGUAGE = 'en'

TOKEN_TYPE = 'stemmers'

# Load language model
kenlm_model = kenlm.Model(full_path('../Resources/languagemodel/lm_giga_20k_vp_3gram.klm'))

# Load stopwords
STOPWORDS_EN = read_lines(full_path('../Resources/stopwords/en.txt'))
STOPWORDS_VI = read_lines(full_path('../Resources/stopwords/vi.txt'))

# Load Stemmer
SNOWBALL_STEMMER = SnowballStemmer('english')


def kenlm_score(sentence):
    num_words = len(remove_punctuation(sentence.split(' ')))
    return kenlm_model.score(sentence) / (num_words + 1)


def eval_linguistic(clusters):
    for cluster in clusters:
        for sentence in cluster:
            sentence['linguistic_score'] = 1. / (1 - kenlm_score(sentence['sentence']))
    return clusters


def eval_informativeness(clusters, lang='en'):
    for cluster in clusters:
        raw_doc = []
        for sentence in cluster:
            stemmers = []
            parsed_sentences = parse(sentence['sentence'], lang)
            for parsed_sentence in parsed_sentences:
                stemmers.extend([SNOWBALL_STEMMER.stem(token) for token in
                                 parsed_sentence['tokens']] if lang is 'en' else parsed_sentence['tokens'])

            sentence['stemmers'] = normalize_word_suffix(' '.join(remove_punctuation(stemmers)), lang=lang)
            raw_doc.append(sentence['stemmers'])

        if len(raw_doc) > 0:
            tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=STOPWORDS_EN if lang is 'en' else STOPWORDS_VI)
            tfidf_matrix = tfidf_vectorizer.fit_transform(raw_doc)
            cosine_similarities = tfidf_matrix * tfidf_matrix.T

            graph = nx.from_scipy_sparse_matrix(cosine_similarities)
            informativeness_score = nx.pagerank(graph)
            for i, sentence in enumerate(cluster):
                sentence['informativeness_score'] = informativeness_score[i]

    return clusters


# Done
def clustering_sentences(docs, token_type='stemmers', sim_threshold=0.5, lang='en'):
    num_docs = len(docs)
    debug('- Number of documents: %d' % num_docs)

    # Determine important document
    merged_docs = []
    raw_docs = []
    for doc in docs:
        raw_doc = []
        for sentence in doc:
            raw_doc.append(' '.join(remove_punctuation(sentence[token_type])))

        raw_doc = ' '.join(raw_doc)

        # Fix for English
        raw_doc = normalize_word_suffix(raw_doc, lang)
        # debug(raw_doc)  # OK

        merged_docs.append(raw_doc)
        raw_docs.append(raw_doc)

    raw_docs.append(' '.join(merged_docs))
    # debug(raw_docs)  # OK

    # Compute cosine similarities and get index of important document
    tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=STOPWORDS_EN if lang is 'en' else STOPWORDS_VI)
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_docs)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # debug(cosine_similarities)  # OK
    imp_doc_idx = cosine_similarities[num_docs:, :num_docs].argmax()
    debug('- Document id: %d\n' % imp_doc_idx, cosine_similarities[num_docs:, :num_docs])

    # Generate clusters
    clusters = []
    raw_sentences = []
    for sentence in docs[imp_doc_idx]:
        clusters.append([
            {
                'doc_name': sentence['doc_name'],
                'pos': sentence['pos'],
                'tokens': sentence['tokens'],
                'tags': sentence['tags']
            }
        ])
        raw_sentences.append(normalize_word_suffix(' '.join(remove_punctuation(sentence[token_type])), lang))

    num_clusters = len(clusters)
    debug('- Number of clusters: %d' % num_clusters)  # OK

    # Align sentences in other documents into clusters
    for i, doc in enumerate(docs):
        if i == imp_doc_idx:
            continue

        for sentence in doc:
            sim_sentences = list(raw_sentences)
            sim_sentences.append(normalize_word_suffix(' '.join(remove_punctuation(sentence[token_type])), lang))

            # Compute cosine similarities and get index of cluster
            tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=STOPWORDS_EN if lang is 'en' else STOPWORDS_VI)
            tfidf_matrix = tfidf_vectorizer.fit_transform(sim_sentences)
            cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
            cosine_similarities = cosine_similarities[num_clusters:, :num_clusters]

            max_sim = cosine_similarities.max()
            cluster_idx = cosine_similarities.argmax()

            # debug(max_sim, cluster_idx, cosine_similarities)  # OK

            if max_sim > sim_threshold:
                clusters[cluster_idx].append({
                    'doc_name': sentence['doc_name'],
                    'pos': sentence['pos'],
                    'tokens': sentence['tokens'],
                    'tags': sentence['tags']
                })

    final_clusters = []
    for cluster in clusters:
        if len(cluster) >= math.ceil(num_docs / 2.):
            final_clusters.append(cluster)

    # debug(json.dumps(final_clusters))  # OK

    return final_clusters


def ordering_clusters(clusters):
    for cluster in clusters:
        debug(cluster)
    return clusters


def compress_clusters(clusters, lang='en'):
    compressed_clusters = []

    for cluster in clusters:
        compressed_cluster = []
        raw_sentences = []
        for sentence in cluster:
            raw_sentences.append(
                ' '.join(['%s/%s' % (token, sentence['tags'][i]) for i, token in enumerate(sentence['tokens'])]))

        compresser = WordGraph(raw_sentences, lang=lang)

        # Get the 200 best paths
        candidates = compresser.get_compression(nb_candidates=100)

        # Bonus
        # reranker = KeyphraseReranker(raw_sentences, candidates, lang=lang)
        # candidates = reranker.rerank_nbest_compressions()

        for cummulative_score, candidate in candidates:
            # Normalize path score by path length
            num_words = len(remove_punctuation([token[0] for token in candidate]))
            normalized_score = cummulative_score / num_words

            compressed_cluster.append({
                'rank': normalized_score,
                'num_words': num_words,
                'sentence': ' '.join([regex.sub('_+', ' ', token[0]) for token in candidate])
            })

        compressed_clusters.append(compressed_cluster)

    return compressed_clusters


def solve_ilp(clusters, num_words=100, lang='en'):
    ilp_problem = LpProblem("mip1", LpMaximize)

    ilp_vars_matrix = []
    raw_sentences = []
    obj_constraints = []
    max_length_constraints = []
    for i, cluster in enumerate(clusters):
        ilp_vars = []
        raw_sentence = []
        max_sentence_constraints = []

        for j, sentence in enumerate(cluster):
            var = LpVariable('%d_%d' % (i, j), cat=LpBinary)
            ilp_vars.append(var)

            raw_sentence.append(sentence['sentence'])

            max_length_constraints.append(sentence['num_words'] * var)

            obj_constraints.append(
                (1. / sentence['num_words']) * sentence['informativeness_score'] * sentence['linguistic_score'] * var)

            max_sentence_constraints.append(var)

        raw_sentences.append(raw_sentence)
        ilp_vars_matrix.append(ilp_vars)
        ilp_problem += lpSum(max_sentence_constraints) <= 1.0

    ilp_problem += lpSum(max_length_constraints) <= num_words
    ilp_problem += lpSum(obj_constraints)

    for i, cluster in enumerate(clusters):
        for j, sentence in enumerate(cluster):
            for _i, _cluster in enumerate(clusters):
                for _j, _sentence in enumerate(_cluster):
                    if i == _i and j == _j:
                        continue
                    # Compute cosine similarities
                    tfidf_vectorizer = TfidfVectorizer(min_df=0,
                                                       stop_words=STOPWORDS_EN if lang is 'en' else STOPWORDS_VI)
                    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence['stemmers'], _sentence['stemmers']])
                    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
                    if cosine_similarities[0][1] >= 0.5:
                        ilp_problem += lpSum([ilp_vars_matrix[i][j], ilp_vars_matrix[_i][_j]]) <= 1.0

    ilp_problem.solve()

    final_sentences = []
    for ilp_var in ilp_problem.variables():
        if ilp_var.varValue == 1.0:
            indices = ilp_var.name.split('_')
            final_sentences.append(raw_sentences[int(indices[0])][int(indices[1])])

    return final_sentences


def main():
    samples = read_json(full_path('../Temp/datasets/%s/info.json' % LANGUAGE))
    for sample in samples:
        debug('Cluster: %s (%s)' % (sample['cluster_name'], sample['original_cluster_name']))

        # Prepare docs in cluster
        raw_docs = sample['docs']
        debug('Parsing documents...')
        parsed_docs = parse_docs(raw_docs=raw_docs, stemmer=SNOWBALL_STEMMER, lang=LANGUAGE)
        debug('Done.')

        # Clustering sentences into clusters
        debug('Clustering sentences in documents...')
        clusters = clustering_sentences(docs=parsed_docs, token_type=TOKEN_TYPE, sim_threshold=0.25, lang=LANGUAGE)
        debug('Done.')

        debug('Ordering clusters...')
        clusters = ordering_clusters(clusters)
        debug('Done.')

        debug('Compressing sentences in each cluster...')
        # compressed_clusters = compress_clusters(clusters, LANGUAGE)
        debug('Done.')
        # scored_clusters = eval_linguistic(clusters=compressed_clusters)
        # scored_clusters = eval_informativeness(clusters=scored_clusters, lang=LANGUAGE)
        # final_sentences = solve_ilp(scored_clusters, lang=LANGUAGE)
        #
        # # debug(' '.join(final_sentences))
        # write_file(' '.join(final_sentences), sample['save'])
        break


def prepare_datasets():
    debug('Preprocessing DUC 2004 dataset...')
    preprocess_duc04('../Datasets/DUC04', '../Temp/datasets/en')
    debug('Done.')
    debug('Preprocessing Vietnamese MDS dataset...')
    preprocess_vimds('../Datasets/VietnameseMDS', '../Temp/datasets/vi')
    debug('Done.')


if __name__ == '__main__':
    if PREPARE_DATASET:
        prepare_datasets()
    else:
        main()
