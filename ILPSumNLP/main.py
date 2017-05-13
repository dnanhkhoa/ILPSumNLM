#!/usr/bin/python
# -*- coding: utf8 -*-
import kenlm

import networkx as nx
from gensim.models import KeyedVectors
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

MAX_WORD_LENGTH = 140 if LANGUAGE is 'en' else 300

TOKEN_TYPE = 'stemmers'

# Load language model
debug('Loading language model...')
LANGUAGE_MODEL_FILES = ['lm_giga_20k_vp_3gram.klm', 'en-3gram.klm', 'en-5gram.klm', 'vi-3gram.klm', 'vi-5gram.klm']
KENLM_MODEL = kenlm.Model(full_path('../Resources/languagemodel/%s' % LANGUAGE_MODEL_FILES[0]))
debug('Done.')

# Load word2vec model
debug('Loading word2vec model...')
WORD2VEC_MODEL_FILES = ['GoogleNews-vectors-negative300.bin', 'W2VModelVN.bin']
WORD2VEC_MODEL = KeyedVectors.load_word2vec_format('../Resources/word2vec/%s' % WORD2VEC_MODEL_FILES[0], binary=True)
debug('Done.')

# Load stopwords
debug('Loading stopwords...')
STOPWORDS_EN = read_lines(full_path('../Resources/stopwords/en.txt'))
STOPWORDS_VI = read_lines(full_path('../Resources/stopwords/vi.txt'))
debug('Done.')

# Load Stemmer
debug('Loading stemmer...')
SNOWBALL_STEMMER = SnowballStemmer('english')
debug('Done.')


# Done
def word2vec_similarity(s1, s2):
    if s1 == s2:
        return 1.0

    s1_tokens = s1.split(' ')
    s2_tokens = s2.split(' ')
    s1_filtered_tokens = set(s1_tokens)
    s2_filtered_tokens = set(s2_tokens)

    if len(s1_filtered_tokens & s2_filtered_tokens) == 0:
        return 0.0

    vocab = WORD2VEC_MODEL.vocab

    for token in s1_filtered_tokens:
        if token not in vocab:
            debug('S1: ', token)
            s1_tokens.remove(token)

    for token in s2_filtered_tokens:
        if token not in vocab:
            debug('S2: ', token)
            s2_tokens.remove(token)

    debug(s1_tokens, s2_tokens)

    return WORD2VEC_MODEL.n_similarity(s1_tokens, s2_tokens)


# Done
def kenlm_score(sentence):
    return KENLM_MODEL.score(sentence) / (len(remove_punctuation(sentence.split(' '))) + 1)


# Done
def eval_linguistic(clusters):
    for cluster in clusters:
        for sentence in cluster:
            sentence['linguistic_score'] = 1. / (1 - kenlm_score(sentence['sentence']))
    return clusters


# Done
def eval_informativeness(clusters, token_type='stemmers', lang='en'):
    for cluster in clusters:
        raw_doc = []
        for sentence in cluster:
            tokens = []
            lemmas = []
            stemmers = []
            parsed_sentences = parse(sentence['sentence'], lang)
            for parsed_sentence in parsed_sentences:
                tokens.extend(parsed_sentence['tokens'])
                lemmas.extend(parsed_sentence['lemmas'] if lang is 'en' else parsed_sentence['tokens'])
                stemmers.extend([SNOWBALL_STEMMER.stem(token) for token in
                                 parsed_sentence['tokens']] if lang is 'en' else parsed_sentence['tokens'])

            sentence['tokens'] = normalize_word_suffix(' '.join(remove_punctuation(tokens)), lang=lang)
            sentence['lemmas'] = normalize_word_suffix(' '.join(remove_punctuation(lemmas)), lang=lang)
            sentence['stemmers'] = normalize_word_suffix(' '.join(remove_punctuation(stemmers)), lang=lang)
            raw_doc.append(sentence[token_type])

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
    debug(json.dumps(raw_docs))  # OK

    # Compute cosine similarities and get index of important document
    tfidf_vectorizer = TfidfVectorizer(min_df=0, stop_words=STOPWORDS_EN if lang is 'en' else STOPWORDS_VI)
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_docs)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # debug(cosine_similarities)  # OK
    imp_doc_idx = cosine_similarities[num_docs:, :num_docs].argmax()
    debug('- Important document: %d\n' % imp_doc_idx, cosine_similarities[num_docs:, :num_docs])

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
        if len(cluster) >= (num_docs // 2):
            final_clusters.append(cluster)

    # debug(json.dumps(final_clusters))  # OK
    debug('- Number of clusters after filtering: %d' % len(final_clusters))  # OK
    for i, cluster in enumerate(final_clusters):
        debug('-- Cluster %d: %d sentences' % (i, len(cluster)))

    return final_clusters


# Done
def ordering_clusters(clusters):
    edges = []
    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            if i == j:
                continue

            cij = 0
            cji = 0
            for sentence_i in cluster_i:
                for sentence_j in cluster_j:
                    if sentence_i['doc_name'] == sentence_j['doc_name']:  # In the same document
                        if sentence_i['pos'] < sentence_j['pos']:  # Cluster i precedes cluster j
                            cij += 1
                        elif sentence_i['pos'] > sentence_j['pos']:  # Cluster j precedes cluster i
                            cji += 1

            edges.append((i, j, cij))
            edges.append((j, i, cji))

    if len(edges) == 0:
        return clusters

    # Find optimal path
    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(edges)

    def compute_node_weights(graph):
        values = {}
        for v in graph:
            values[v] = graph.out_degree(v, 'weight') - graph.in_degree(v, 'weight')
        return values

    cluster_order = []
    while digraph.number_of_nodes() > 0:
        nodes = compute_node_weights(graph=digraph)
        # Using the original order if there is more max values
        node = max(nodes, key=nodes.get)
        cluster_order.append((node, nodes[node]))
        digraph.remove_node(node)

    debug('- Order:', cluster_order)

    final_clusters = []
    for i, _ in cluster_order:
        final_clusters.append(clusters[i])

    # debug('- Before:', json.dumps(clusters))
    # debug('- After:', json.dumps(final_clusters))

    return final_clusters


# Done
def compress_clusters(clusters, num_words=8, num_candidates=200, lang='en'):
    compressed_clusters = []

    for cluster in clusters:
        raw_sentences = []
        for sentence in cluster:
            raw_sentences.append(
                ' '.join(['%s/%s' % (token, sentence['tags'][i]) for i, token in enumerate(sentence['tokens'])]))

        # debug(json.dumps(raw_sentences))  # OK
        compresser = WordGraph(raw_sentences, nb_words=num_words, lang=lang)

        # Get the 200 best paths
        candidates = compresser.get_compression(nb_candidates=num_candidates)

        # Use Keyphrase reranking method
        # reranker = KeyphraseReranker(raw_sentences, candidates, lang=lang)
        # candidates = reranker.rerank_nbest_compressions()

        compressed_cluster = []

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


def solve_ilp(clusters, token_type='stemmers', num_words=100, lang='en'):
    ilp_problem = LpProblem("amds", LpMaximize)

    raw_sentences = []

    ilp_vars_matrix = []

    obj_constraint = []
    max_length_constraint = []

    for i, cluster in enumerate(clusters):
        ilp_vars = []
        raw_sentence = []
        max_sentence_constraints = []

        for j, sentence in enumerate(cluster):
            var = LpVariable('%d_%d' % (i, j), cat=LpBinary)
            ilp_vars.append(var)

            raw_sentence.append(sentence['sentence'])

            max_length_constraint.append(sentence['num_words'] * var)

            obj_constraint.append(
                (1. / sentence['num_words']) * sentence['informativeness_score'] * sentence['linguistic_score'] * var)

            max_sentence_constraints.append(var)

        raw_sentences.append(raw_sentence)
        ilp_vars_matrix.append(ilp_vars)
        ilp_problem += lpSum(max_sentence_constraints) <= 1.0

    ilp_problem += lpSum(max_length_constraint) <= num_words
    ilp_problem += lpSum(obj_constraint)

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
    s1 = 'An Apple'
    s2 = 'A apple'
    debug(word2vec_similarity(s1, s2))

    return
    samples = read_json(full_path('../Temp/datasets/%s/info.json' % LANGUAGE))
    for sample in samples:
        sample = samples[0]

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

        # debug('Ordering clusters...')
        # clusters = ordering_clusters(clusters=clusters)
        # debug('Done.')
        #
        # debug('Compressing sentences in each cluster...')
        # compressed_clusters = compress_clusters(clusters=clusters, num_words=8, num_candidates=200, lang=LANGUAGE)
        # debug('Done.')
        #
        # debug('Computing linguistic score...')
        # scored_clusters = eval_linguistic(clusters=compressed_clusters)
        # debug('Done.')
        #
        # debug('Computing informativeness score...')
        # scored_clusters = eval_informativeness(clusters=scored_clusters, token_type=TOKEN_TYPE, lang=LANGUAGE)
        # debug('Done.')
        #
        # debug('Solving ILP...')
        # final_sentences = []
        # from absummarizer import Example
        # for cluster in clusters:
        #     raw_doc = []
        #     for sentence in cluster:
        #         raw_doc.append(' '.join(sentence['tokens']))
        #
        #     final_sentence = Example.segmentize(' '.join(raw_doc))
        #     final_sentences.append(Example.generateSummaries(final_sentence, mode="Abstractive"))
        #
        # # final_sentences = solve_ilp(clusters=scored_clusters, token_type=TOKEN_TYPE, num_words=140, lang=LANGUAGE)
        # debug('Done.')
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
