#!/usr/bin/python
# -*- coding: utf8 -*-
import kenlm
import timeit
from datetime import datetime
from random import shuffle

import networkx as nx
from nltk.stem import SnowballStemmer
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum
from pyemd import emd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from core import WordGraph, np
from utils.preprocessing import *
from utils.utils import *

# Configs ============================================================

PREPARE_DATASET = False

# Define dataset language
LANGUAGE = 'en'

# Define methods
SIMILARITY_METHOD = ['freq', 'w2v', 'wmd', 'd2v'][0]

USE_TFIDF = False

TOKEN_TYPE = ['tokens', 'lemmas', 'stemmers'][0 if SIMILARITY_METHOD != 'freq' else 2]

LANGUAGE_MODEL_METHODS = ['ngram', 'rmn'][0]

# Load language model | OK
KENLM_MODEL = kenlm.Model(
    full_path('../Resources/languagemodel/%s-3gram.klm' % LANGUAGE)) if LANGUAGE_MODEL_METHODS is 'ngram' else None

# Define vectorizer | OK
VECTORIZER = TfidfVectorizer if USE_TFIDF else CountVectorizer

# Load word2vec model | OK
WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format('../Resources/word2vec/%s.bin' % LANGUAGE,
                                                                 binary=True) if SIMILARITY_METHOD in ['w2v',
                                                                                                       'wmd'] else None

# Load doc2vec model | OK
DOC2VEC_MODEL = gensim.models.Doc2Vec.load(
    '../Resources/doc2vec/%s/model.bin' % LANGUAGE) if SIMILARITY_METHOD == 'd2v' else None

# Load stopwords | OK
STOPWORDS = read_lines(full_path('../Resources/stopwords/%s.txt' % LANGUAGE))

# Load Stemmer | OK
SNOWBALL_STEMMER = SnowballStemmer('english')


# =====================================================================


# Done
def word2vec_similarity(d1, d2):
    d1 = d1.lower()
    d2 = d2.lower()

    if d1 == d2:
        return 1.0

    s1_tokens = d1.split(' ')
    s2_tokens = d2.split(' ')
    s1_filtered_tokens = set(s1_tokens)
    s2_filtered_tokens = set(s2_tokens)

    # No words
    if len(s1_filtered_tokens & s2_filtered_tokens) == 0:
        return 0.0

    s1_invalid_tokens = []
    for token in s1_filtered_tokens:
        if token not in WORD2VEC_MODEL.vocab:
            s1_invalid_tokens.append(token)

    s1_tokens = list(filter(lambda x: x not in s1_invalid_tokens, s1_tokens))

    s2_invalid_tokens = []
    for token in s2_filtered_tokens:
        if token not in WORD2VEC_MODEL.vocab:
            s2_invalid_tokens.append(token)

    s2_tokens = list(filter(lambda x: x not in s2_invalid_tokens, s2_tokens))

    # No words
    if len(s1_tokens) == 0 or len(s2_tokens) == 0:
        return 0.0

    return WORD2VEC_MODEL.n_similarity(s1_tokens, s2_tokens)


# Done
def wmd_similarity(d1, d2):
    d1 = d1.lower()
    d2 = d2.lower()

    if d1 == d2:
        return 1.0

    s1_tokens = d1.split(' ')
    s2_tokens = d2.split(' ')
    s1_filtered_tokens = set(s1_tokens)
    s2_filtered_tokens = set(s2_tokens)

    if len(s1_filtered_tokens & s2_filtered_tokens) == 0:
        return 0.0

    vocab = [word for word in (s1_filtered_tokens | s2_filtered_tokens) if word in WORD2VEC_MODEL.vocab]

    vectorizer = VECTORIZER(vocabulary=vocab).fit([d1, d2])
    W_ = np.array([WORD2VEC_MODEL[word] for word in vectorizer.get_feature_names() if word in WORD2VEC_MODEL])
    D_ = euclidean_distances(W_)
    D_ = D_.astype(np.double)
    D_ /= D_.max()

    v_1, v_2 = vectorizer.transform([d1, d2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()

    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    return 1.0 - emd(v_1, v_2, D_)


# Done
def get_d2v_vector(doc):
    start_alpha = 0.01
    infer_epoch = 1000
    return DOC2VEC_MODEL.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)


# Done
def doc_similarity(docs, method=SIMILARITY_METHOD):
    size = len(docs)
    cosine_similarities = None
    if method == 'w2v':
        cosine_similarities = np.ones((size, size))
        for i in range(0, size - 1):
            for j in range(i + 1, size):
                cosine_similarities[i][j] = cosine_similarities[j][i] = word2vec_similarity(docs[i], docs[j])
    elif method == 'wmd':
        cosine_similarities = np.ones((size, size))
        for i in range(0, size - 1):
            for j in range(i + 1, size):
                cosine_similarities[i][j] = cosine_similarities[j][i] = wmd_similarity(docs[i], docs[j])
    elif method == 'd2v':
        pass
    else:
        vectorizer = VECTORIZER(min_df=0, stop_words=STOPWORDS)
        matrix = vectorizer.fit_transform(docs)
        cosine_similarities = cosine_similarity(matrix, matrix)

    return cosine_similarities


# Done
def kenlm_score(sentence):
    tokens = remove_punctuation(sentence.split(' '))
    return KENLM_MODEL.score(' '.join(tokens)) / (1. + len(tokens))


# Done
def eval_linguistic(clusters):
    for cluster in clusters:
        for sentence in cluster:
            sentence['linguistic_score'] = 1. / (1. - kenlm_score(sentence['sentence']))
    return clusters


# Done
def eval_informativeness(clusters):
    for cluster in clusters:
        raw_doc = []
        for sentence in cluster:
            raw_doc.append(normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))

        if len(raw_doc) > 0:
            graph = nx.from_numpy_matrix(doc_similarity(raw_doc), create_using=nx.DiGraph())
            informativeness_score = nx.pagerank(graph)
            for i, sentence in enumerate(cluster):
                sentence['informativeness_score'] = informativeness_score[i]

    return clusters


# Done
def clustering_sentences(docs, cluster_threshold=1, sim_threshold=0.5, n_top=None):
    num_docs = len(docs)
    debug('- Number of documents: %d' % num_docs)

    # Determine important document
    raw_docs = []
    for doc in docs:
        raw_doc = []
        for sentence in doc:
            raw_doc.append(' '.join(remove_punctuation(sentence[TOKEN_TYPE])))

        # Fix for English
        raw_doc = normalize_word_suffix(' '.join(raw_doc), lang=LANGUAGE)
        # debug(raw_doc)  # OK

        raw_docs.append(raw_doc)

    raw_docs.append(' '.join(raw_docs))
    # debug(json.dumps(raw_docs))  # OK

    # Compute cosine similarities and get index of important document
    imp_doc_idx = doc_similarity(raw_docs)[num_docs:, :num_docs].argmax()
    debug('- Important document: %d' % imp_doc_idx)

    # Generate clusters
    clusters = []
    raw_sentences = []
    for sentence in docs[imp_doc_idx]:
        sentence['sim'] = 1.0
        clusters.append([sentence])
        raw_sentences.append(normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))

    num_clusters = len(clusters)
    debug('- Number of clusters: %d' % num_clusters)  # OK

    # Align sentences in other documents into clusters
    sentence_mapping = []
    for i, doc in enumerate(docs):
        if i == imp_doc_idx:
            continue

        for sentence in doc:
            sentence_mapping.append(sentence)
            raw_sentences.append(
                normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))

    # Compute cosine similarities and get index of cluster
    cosine_similarities = doc_similarity(raw_sentences)[num_clusters:, :num_clusters]
    for i, row in enumerate(cosine_similarities):
        max_sim = row.max()
        if max_sim >= sim_threshold:
            sentence = sentence_mapping[i]
            sentence['sim'] = max_sim
            clusters[row.argmax()].append(sentence)

    ordered_clusters = []
    if n_top is None:
        ordered_clusters = clusters
    else:
        for cluster in clusters:
            ordered_cluster = sorted(cluster, key=lambda s: s['sim'], reverse=True)
            ordered_clusters.append(ordered_cluster[:n_top])

    final_clusters = []
    for cluster in ordered_clusters:
        if len(cluster) >= cluster_threshold:
            final_clusters.append(cluster)

    # debug(json.dumps(final_clusters))  # OK
    debug('- Number of clusters after filtering: %d' % len(final_clusters))  # OK
    for i, cluster in enumerate(final_clusters):
        debug('-- Cluster %d: %d sentences' % (i + 1, len(cluster)))

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

    return final_clusters


# Done
def remove_similar_sentences(compressed_clusters, original_clusters, sim_threshold=0.8):
    final_clusters = []
    for i, cluster in enumerate(original_clusters):
        num_original_sentences = len(cluster)
        raw_doc = []
        for sentence in cluster:
            raw_doc.append(normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))
        for sentence in compressed_clusters[i]:
            raw_doc.append(normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))

        final_cluster = []
        cosine_similarities = doc_similarity(raw_doc)[num_original_sentences:, :num_original_sentences]
        for j, row in enumerate(cosine_similarities):
            if row.max() < sim_threshold:
                final_cluster.append(compressed_clusters[i][j])

        if len(final_cluster) > 0:
            final_clusters.append(final_cluster)

    return final_clusters


# Done
def compress_clusters(clusters, num_words=8, max_words=30, num_candidates=200, sim_threshold=0.8, simple_method=True):
    compressed_clusters = []

    for cluster in clusters:
        raw_sentences = []
        for sentence in cluster:
            raw_sentences.append(
                ' '.join(['%s/%s' % (token, sentence['tags'][i]) for i, token in enumerate(sentence['tokens'])]))

        # debug(json.dumps(raw_sentences))  # OK
        compresser = WordGraph(sentence_list=raw_sentences, stopwords=STOPWORDS, nb_words=num_words, lang=LANGUAGE)

        if simple_method:
            # Get simple paths
            graph = nx.convert_node_labels_to_integers(compresser.graph)
            nodes = graph.nodes(data=True)

            labels = []
            start_node = None
            end_node = None
            for node_id, node_attr in nodes:
                if node_attr['label'] == compresser.start:
                    start_node = node_id
                elif node_attr['label'] == compresser.stop:
                    end_node = node_id

                labels.append(node_attr['label'])

            all_simple_paths = []
            all_simple_paths_iter = nx.all_simple_paths(graph, source=start_node, target=end_node, cutoff=max_words)
            for simple_path in all_simple_paths_iter:
                all_simple_paths.append(simple_path)
                if len(all_simple_paths) >= 100000:
                    break

            sentence_container = {}
            candidates = []

            for simple_path in all_simple_paths:
                tokens = []

                paired_parentheses = 0
                quotation_mark_number = 0
                for node in simple_path:
                    word = labels[node]

                    if word == compresser.start or word == compresser.stop:
                        continue

                    if word == '(':
                        paired_parentheses -= 1
                    elif word == ')':
                        paired_parentheses += 1
                    elif word == '"':
                        quotation_mark_number += 1

                    tokens.append(labels[node])

                raw_sentence = ' '.join(tokens)
                sentence_length = len(remove_punctuation(tokens))
                if num_words <= sentence_length <= max_words and paired_parentheses == 0 and (
                            quotation_mark_number % 2) == 0 and raw_sentence not in sentence_container:
                    candidates.append(tokens)
                    sentence_container[raw_sentence] = True

            shuffle(candidates)
            candidates = candidates[:num_candidates]

            compressed_cluster = []
            for candidate in candidates:
                compressed_cluster.append({
                    'num_words': len(remove_punctuation(candidate)),
                    'sentence': ' '.join([regex.sub('_+', ' ', token) for token in candidate])
                })

            compressed_clusters.append(compressed_cluster)
        else:
            candidates = compresser.get_compression(nb_candidates=num_candidates)

            compressed_cluster = []
            for score, candidate in candidates:
                sentence_length = len(remove_punctuation([token[0] for token in candidate]))

                if sentence_length > max_words:
                    continue

                compressed_cluster.append({
                    'num_words': sentence_length,
                    'score': score / sentence_length,
                    'sentence': ' '.join([regex.sub('_+', ' ', token[0]) for token in candidate])
                })

            compressed_clusters.append(sorted(compressed_cluster, key=lambda s: s['score']))

    for cluster in compressed_clusters:
        for sentence in cluster:
            tokens = []
            lemmas = []
            stemmers = []
            parsed_sentences = parse(sentence['sentence'], lang=LANGUAGE)
            for parsed_sentence in parsed_sentences:
                tokens.extend(parsed_sentence['tokens'])
                lemmas.extend(parsed_sentence['lemmas'] if LANGUAGE is 'en' else parsed_sentence['tokens'])
                stemmers.extend([SNOWBALL_STEMMER.stem(token) for token in
                                 parsed_sentence['tokens']] if LANGUAGE is 'en' else parsed_sentence['tokens'])

            sentence['tokens'] = tokens
            sentence['lemmas'] = lemmas
            sentence['stemmers'] = stemmers

    debug('- Number of sentences in each cluster:')
    for i, cluster in enumerate(compressed_clusters):
        debug('-- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    compressed_clusters = remove_similar_sentences(compressed_clusters, clusters, sim_threshold=sim_threshold)

    debug('- Number of sentences in each cluster after removing similar sentences:')
    for i, cluster in enumerate(compressed_clusters):
        debug('-- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    return compressed_clusters


# Done
def solve_ilp(clusters, num_words=150, sim_threshold=0.5, greedy_mode=False):
    final_sentences = []

    if greedy_mode:

        # Greedy mode
        selected_sentences = []
        summary_length = 0

        # Sort clusters by number of sentences in descending order
        cluster_indices = sorted([(i, len(cluster)) for i, cluster in enumerate(clusters)], key=lambda v: v[1],
                                 reverse=True)

        debug(cluster_indices)

        for cluster in clusters:
            ilp_problem = LpProblem("amds", LpMaximize)

            ilp_vars = []

            obj_constraint = []
            max_sentence_constraint = []

            for i, sentence in enumerate(cluster):
                var = LpVariable(str(i), cat=LpBinary)
                ilp_vars.append(var)

                obj_constraint.append(
                    (1. / sentence['num_words']) * sentence['informativeness_score'] * sentence[
                        'linguistic_score'] * var)

                max_sentence_constraint.append(var)

                ilp_problem += summary_length + sentence['num_words'] * var <= num_words

            if len(selected_sentences) > 0:
                raw_sentences = []
                for sentence in selected_sentences:
                    raw_sentences.append(
                        normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))
                for sentence in cluster:
                    raw_sentences.append(
                        normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE))

                # Compute cosine similarities
                num_selected_sentences = len(selected_sentences)
                cosine_similarities = doc_similarity(raw_sentences)[num_selected_sentences:, :num_selected_sentences]
                for k, row in enumerate(cosine_similarities):
                    if row.max() >= sim_threshold:
                        ilp_problem += ilp_vars[k] == 0.0

            ilp_problem += lpSum(max_sentence_constraint) <= 1.0
            ilp_problem += lpSum(obj_constraint)

            ilp_problem.solve()

            for ilp_var in ilp_problem.variables():
                if ilp_var.varValue == 1.0:
                    i = ilp_var.name
                    summary_length += cluster[int(i)]['num_words']
                    selected_sentences.append(cluster[int(i)])

        for sentence in selected_sentences:
            final_sentences.append(sentence['sentence'])
    else:
        # Standard mode
        ilp_problem = LpProblem("amds", LpMaximize)

        ilp_vars_table = []

        obj_constraint = []
        max_length_constraint = []

        for i, cluster in enumerate(clusters):
            ilp_vars = []
            max_sentence_constraint = []

            for j, sentence in enumerate(cluster):
                var = LpVariable('%d_%d' % (i, j), cat=LpBinary)
                ilp_vars.append(var)

                max_sentence_constraint.append(var)
                max_length_constraint.append(sentence['num_words'] * var)

                obj_constraint.append(
                    (1. / sentence['num_words']) * sentence['informativeness_score'] * sentence[
                        'linguistic_score'] * var)

            ilp_vars_table.append(ilp_vars)

            ilp_problem += lpSum(max_sentence_constraint) <= 1.0

        ilp_problem += lpSum(max_length_constraint) <= num_words
        ilp_problem += lpSum(obj_constraint)

        for i, cluster in enumerate(clusters):
            for _i, _cluster in enumerate(clusters):
                if i == _i:
                    continue
                for j, sentence in enumerate(cluster):
                    for _j, _sentence in enumerate(_cluster):
                        # Compute cosine similarities
                        cosine_similarities = doc_similarity([
                            normalize_word_suffix(' '.join(remove_punctuation(sentence[TOKEN_TYPE])), lang=LANGUAGE),
                            normalize_word_suffix(' '.join(remove_punctuation(_sentence[TOKEN_TYPE])), lang=LANGUAGE)
                        ])
                        if cosine_similarities[0][1] >= sim_threshold:
                            ilp_problem += lpSum([ilp_vars_table[i][j], ilp_vars_table[_i][_j]]) <= 1.0

        ilp_problem.solve()

        for ilp_var in ilp_problem.variables():
            if ilp_var.varValue == 1.0:
                i, j = ilp_var.name.split('_')
                final_sentences.append(clusters[int(i)][int(j)]['sentence'])

    return final_sentences


def main():
    start_time = timeit.default_timer()

    backup = []
    samples = read_json(full_path('../Temp/datasets/%s/info.json' % LANGUAGE))
    for sample in samples:
        num_docs = len(sample['docs'])

        debug('Cluster: %s (%s)' % (sample['cluster_name'], sample['original_cluster_name']))

        num_words = 120
        # for model in sample['models']:
        #     num_words = max(num_words, model['num_words'])
        # num_words = round(num_words / 10.) * 10

        # Prepare docs in cluster
        raw_docs = sample['docs']
        debug('Parsing documents...')
        parsed_docs = parse_docs(raw_docs=raw_docs, stemmer=SNOWBALL_STEMMER, lang=LANGUAGE)
        debug('Done.')

        # Clustering sentences into clusters
        debug('Clustering sentences in documents...')
        clusters = clustering_sentences(docs=parsed_docs, cluster_threshold=3, sim_threshold=0.15, n_top=30)
        debug('Done.')

        debug('Ordering clusters...')
        clusters = ordering_clusters(clusters=clusters)
        debug('Done.')

        debug('Compressing sentences in each cluster...')
        compressed_clusters = compress_clusters(clusters=clusters, num_words=8, max_words=40, num_candidates=200,
                                                sim_threshold=0.8, simple_method=True)
        debug('Done.')

        backup.append(compressed_clusters)

        debug('Computing linguistic score...')
        scored_clusters = eval_linguistic(clusters=compressed_clusters)
        debug('Done.')

        debug('Computing informativeness score...')
        scored_clusters = eval_informativeness(clusters=scored_clusters)
        debug('Done.')

        debug('Solving ILP...')
        final_sentences = solve_ilp(clusters=scored_clusters, num_words=num_words, sim_threshold=0.5, greedy_mode=True)
        debug('Done.')

        debug(' '.join(final_sentences))
        write_file(' '.join(final_sentences), sample['save'])

    debug('Total time: %d s' % (timeit.default_timer() - start_time))

    # Backup
    write_json(backup, '../Temp/datasets/%s/backup-%s.json' % (LANGUAGE, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))


# Done
def prepare_datasets():
    debug('Preprocessing DUC 2004 dataset...')
    preprocess_duc04('../Datasets/DUC04', '../Temp/datasets/en')
    debug('Done.')
    debug('Preprocessing Vietnamese MDS dataset...')
    preprocess_vimds('../Datasets/VietnameseMDS', '../Temp/datasets/vi')
    debug('Done.')


# Done
if __name__ == '__main__':
    if PREPARE_DATASET:
        prepare_datasets()
    else:
        main()
