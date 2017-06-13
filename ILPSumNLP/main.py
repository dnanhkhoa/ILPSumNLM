#!/usr/bin/python
# -*- coding: utf8 -*-
import Stemmer
import kenlm
import timeit

import networkx as nx
import numpy as np
import pulp
from pyemd import emd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils.preprocessing import *
from utils.utils import *
from wordgraph import WordGraph

# Configs =============================================================

PREPARE_DATASET = False

LANGUAGE = 'en'

TOKEN_TYPE = ['tokens', 'stemmers'][1]

SIMILARITY_METHOD = ['freq', 'w2v', 'wmd'][0]

LANGUAGE_MODEL_METHOD = ['ngram', 'rmn'][0]

USE_STOPWORDS = True

USE_TFIDF = False

# Load stemmer model | OK
ENGLISH_STEMMER = Stemmer.Stemmer('en') if TOKEN_TYPE == 'stemmers' else None


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: ENGLISH_STEMMER.stemWords(analyzer(doc))


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: ENGLISH_STEMMER.stemWords(analyzer(doc))


# Load automatically ==================================================

if LANGUAGE == 'vi' or SIMILARITY_METHOD != 'freq':
    TOKEN_TYPE = 'tokens'

# Load language model | OK
KENLM_MODEL = None

if LANGUAGE_MODEL_METHOD == 'ngram':
    KENLM_MODEL = kenlm.Model(full_path('../Resources/languagemodel/giga-%s-3gram.klm' % LANGUAGE))

# Load word2vec model | OK
WORD2VEC_MODEL = None

if SIMILARITY_METHOD in ['w2v', 'wmd']:
    WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format('../Resources/word2vec/%s.bin' % LANGUAGE,
                                                                     binary=True)

# Load stopwords | OK
STOPWORDS = read_lines(full_path('../Resources/stopwords/%s.txt' % LANGUAGE)) if USE_STOPWORDS else []

# Define vectorizer | OK
VECTORIZER = None

if TOKEN_TYPE == 'stemmers':
    if USE_TFIDF:
        VECTORIZER = StemmedTfidfVectorizer
    else:
        VECTORIZER = StemmedCountVectorizer
else:
    if USE_TFIDF:
        VECTORIZER = TfidfVectorizer
    else:
        VECTORIZER = CountVectorizer

debug(LANGUAGE, TOKEN_TYPE, SIMILARITY_METHOD, LANGUAGE_MODEL_METHOD)
debug(VECTORIZER.__name__)


# =====================================================================

# OK
def word2vec_similarity(d1, d2):
    d1 = d1.lower()
    d2 = d2.lower()

    if d1 == d2:
        return 1.0

    s1_tokens = d1.split(' ')
    s2_tokens = d2.split(' ')
    s1_filtered_tokens = set(s1_tokens)
    s2_filtered_tokens = set(s2_tokens)

    s1_invalid_tokens = []
    for token in s1_filtered_tokens:
        if token not in WORD2VEC_MODEL.vocab or token in STOPWORDS:
            s1_invalid_tokens.append(token)

    s1_tokens = list(filter(lambda x: x not in s1_invalid_tokens, s1_tokens))

    s2_invalid_tokens = []
    for token in s2_filtered_tokens:
        if token not in WORD2VEC_MODEL.vocab or token in STOPWORDS:
            s2_invalid_tokens.append(token)

    s2_tokens = list(filter(lambda x: x not in s2_invalid_tokens, s2_tokens))

    # No words
    if len(s1_tokens) == 0 or len(s2_tokens) == 0:
        return 0.0

    return WORD2VEC_MODEL.n_similarity(s1_tokens, s2_tokens)


def wmd_similarity(d1, d2):
    d1 = d1.lower()
    d2 = d2.lower()

    if d1 == d2:
        return 1.0

    s1_tokens = d1.split(' ')
    s2_tokens = d2.split(' ')
    s1_filtered_tokens = set(s1_tokens)
    s2_filtered_tokens = set(s2_tokens)

    vocab = [word for word in (s1_filtered_tokens | s2_filtered_tokens) if
             word in WORD2VEC_MODEL.vocab and word not in STOPWORDS]

    if len(vocab) == 0:
        return 0.0

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


def sentence_similarity(docs, method=SIMILARITY_METHOD, stopwords=STOPWORDS):
    size = len(docs)
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
    else:
        vectorizer = VECTORIZER(min_df=0, stop_words=STOPWORDS)
        matrix = vectorizer.fit_transform(docs)
        cosine_similarities = cosine_similarity(matrix, matrix)

    return cosine_similarities


# OK
def kenlm_score(sentence):
    return KENLM_MODEL.score(sentence) / (1. + len(sentence.split(' ')))


# OK
def eval_linguistic_score(cluster):
    for sentence in cluster:
        sentence['linguistic_score'] = 1. / (1. - kenlm_score(sentence['sentence']))
    return cluster


# OK
def eval_linguistic_score_clusters(clusters):
    return pool_executor(fn=eval_linguistic_score, args=[clusters], executor_mode=1)


# OK
def eval_informativeness_score(cluster):
    sentences = []
    for sentence in cluster:
        sentences.append(sentence['sentence'])

    if len(sentences) > 0:
        # Undirected graph
        graph = nx.from_numpy_matrix(sentence_similarity(sentences))
        informativeness_score = nx.pagerank(graph)
        for i, sentence in enumerate(cluster):
            sentence['informativeness_score'] = informativeness_score[i]

    return cluster


# OK
def eval_informativeness_score_clusters(clusters):
    return pool_executor(fn=eval_informativeness_score, args=[clusters], executor_mode=1)


def clustering_sentences(docs, cluster_threshold=1, sim_threshold=0.5, n_top_sentences=None, n_top_clusters=None):
    num_docs = len(docs)
    debug('- Number of documents: %d' % num_docs)

    # Determine important document
    raw_docs = []
    for doc in docs:
        raw_doc = []
        for sentence in doc:
            raw_doc.append(' '.join(remove_punctuation(sentence['tokens'])))

        raw_docs.append(normalize_word_suffix(' '.join(raw_doc), lang=LANGUAGE))

    raw_docs.append(' '.join(raw_docs))
    # debug(json.dumps(raw_docs))  # OK

    # Compute cosine similarities and get index of important document
    imp_doc_idx = sentence_similarity(raw_docs)[num_docs:, :num_docs].argmax()
    debug('- Important document: %d' % imp_doc_idx)

    # Generate clusters
    clusters = []
    raw_sentences = []
    for sentence in docs[imp_doc_idx]:
        sentence['sim'] = 1.0
        clusters.append([sentence])
        raw_sentences.append(normalize_word_suffix(' '.join(remove_punctuation(sentence['tokens'])), lang=LANGUAGE))

    num_clusters = len(clusters)
    debug('- Number of clusters: %d' % num_clusters)  # OK

    # Align sentences in other documents into clusters
    sentence_mapping = []
    for i, doc in enumerate(docs):
        if i == imp_doc_idx:  # Skip important document
            continue

        for sentence in doc:
            sentence_mapping.append(sentence)
            raw_sentences.append(
                normalize_word_suffix(' '.join(remove_punctuation(sentence['tokens'])), lang=LANGUAGE))

    # Compute cosine similarities and get index of cluster
    cosine_similarities = sentence_similarity(raw_sentences)[num_clusters:, :num_clusters]
    for i, row in enumerate(cosine_similarities):
        max_sim = row.max()
        if max_sim >= sim_threshold:
            sentence = sentence_mapping[i]
            sentence['sim'] = max_sim
            clusters[row.argmax()].append(sentence)

    ordered_clusters = []
    if n_top_sentences is None:
        ordered_clusters = clusters
    else:
        for cluster in clusters:
            ordered_cluster = sorted(cluster, key=lambda s: s['sim'], reverse=True)
            ordered_clusters.append(ordered_cluster[:n_top_sentences])

    final_clusters = []
    for cluster in ordered_clusters:
        if len(cluster) >= cluster_threshold:
            final_clusters.append(cluster)

    if n_top_clusters is not None:
        # Sort clusters by number of sentences in descending order
        cluster_indices = sorted([(i, len(cluster)) for i, cluster in enumerate(final_clusters)], key=lambda v: v[1],
                                 reverse=True)
        cluster_indices = dict(cluster_indices[:n_top_clusters])
        final_clusters = [cluster for i, cluster in enumerate(final_clusters) if i in cluster_indices]

    # debug(json.dumps(final_clusters))  # OK
    debug('- Number of clusters after filtering: %d' % len(final_clusters))  # OK
    for i, cluster in enumerate(final_clusters):
        debug('-- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    return final_clusters


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


def remove_similar_sentences(compressed_clusters, original_clusters, sim_threshold=0.8):
    final_clusters = []
    for i, cluster in enumerate(original_clusters):
        num_original_sentences = len(cluster)
        raw_doc = []
        for sentence in cluster:
            raw_doc.append(normalize_word_suffix(' '.join(remove_punctuation(sentence['tokens'])), lang=LANGUAGE))
        for sentence in compressed_clusters[i]:
            raw_doc.append(normalize_word_suffix(' '.join(remove_punctuation(sentence['tokens'])), lang=LANGUAGE))

        final_cluster = []
        cosine_similarities = sentence_similarity(raw_doc)[num_original_sentences:, :num_original_sentences]
        for j, row in enumerate(cosine_similarities):
            if row.max() < sim_threshold:
                final_cluster.append(compressed_clusters[i][j])

        if len(final_cluster) > 0:
            final_clusters.append(final_cluster)

    return final_clusters


def compress_clusters(clusters, num_words=8, num_candidates=200, sim_threshold=0.8):
    compressed_clusters = []

    for cluster in clusters:
        raw_sentences = []
        for sentence in cluster:
            raw_sentences.append(
                ' '.join(['%s/%s' % (token, sentence['tags'][i]) for i, token in enumerate(sentence['tokens'])]))

        compresser = WordGraph(sentence_list=raw_sentences, stopwords=STOPWORDS, nb_words=num_words, lang=LANGUAGE)

        candidates = compresser.get_compression(nb_candidates=num_candidates)
        # reranker = KeyphraseReranker(raw_sentences, candidates, lang='en')
        # reranked_candidates = reranker.rerank_nbest_compressions()

        compressed_cluster = []
        for score, candidate in candidates:
            tokens = [token[0] for token in candidate]
            sentence_length = len(remove_punctuation(tokens))

            compressed_cluster.append({
                'num_words': sentence_length,  # Do not include number of punctuations in length of sentence
                'score': score / len(tokens),  # Do not remove the punctuations
                'sentence': ' '.join(tokens),
                'tokens': tokens
            })

        compressed_clusters.append(compressed_cluster)
        # compressed_clusters.append(sorted(compressed_cluster, key=lambda s: s['score']))

    debug('- Number of sentences in each cluster:')
    for i, cluster in enumerate(compressed_clusters):
        debug('-- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    compressed_clusters = remove_similar_sentences(compressed_clusters, clusters, sim_threshold=sim_threshold)

    debug('- Number of sentences in each cluster after removing similar sentences:')
    for i, cluster in enumerate(compressed_clusters):
        debug('-- Cluster %d: %d sentences' % (i + 1, len(cluster)))

    return compressed_clusters


# OK
def solve_ilp(clusters, num_words=100, sim_threshold=0.5):
    # Define problem
    ilp_problem = pulp.LpProblem("ILPSumNLP", pulp.LpMaximize)

    # For storing
    sentences = []
    ilp_vars_matrix = []

    # For creating constraint
    obj_function = []
    length_constraint = []

    # Iterate over the clusters
    for i, cluster in enumerate(clusters):

        ilp_vars = []

        # Iterate over the sentence in the clusters
        for j, sentence in enumerate(cluster):
            var = pulp.LpVariable('var_%d_%d' % (i, j), cat=pulp.LpBinary)

            # Prepare objective function
            obj_function.append(sentence['informativeness_score'] * sentence['linguistic_score'] * var)

            # Prepare constraints
            length_constraint.append(sentence['num_words'] * var)

            # Store ILP variable
            ilp_vars.append(var)

            # Store sentence
            sentences.append(sentence['sentence'])

        # Store ILP variables
        ilp_vars_matrix.append(ilp_vars)

        # Create constraint
        ilp_problem += pulp.lpSum(ilp_vars) <= 1.0, 'Cluster_%d constraint' % i

    # Create constraint
    ilp_problem += pulp.lpSum(length_constraint) <= num_words, 'Length constraint'

    # Create objective function
    ilp_problem += pulp.lpSum(obj_function), 'Objective function'

    # Compute cosine similarities
    cosine_similarities = sentence_similarity(sentences)

    # Filter similar sentence between clusters
    pos = 0
    for i, cluster in enumerate(clusters):
        _pos = 0
        for _i, _cluster in enumerate(clusters):
            if i != _i:
                for j, sentence in enumerate(cluster):
                    for _j, _sentence in enumerate(_cluster):
                        if cosine_similarities[pos + j][_pos + _j] >= sim_threshold:
                            ilp_problem += ilp_vars_matrix[i][j] + ilp_vars_matrix[_i][_j] <= 1.0, \
                                           'Sim(var_%d_%d,var_%d_%d) constraint' % (i, j, _i, _j)

            _pos += len(_cluster)
        pos += len(cluster)

    # Maximizing objective function
    ilp_problem.solve(pulp.GLPK(msg=0))

    # Create summary
    final_sentences = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            if ilp_vars_matrix[i][j].varValue == 1.0:
                final_sentences.append(clusters[i][j]['sentence'])
    return final_sentences


def main():
    test2()
    return

    samples = read_json(full_path('../Temp/datasets/%s/info.json' % LANGUAGE))[0:1]

    backup = []
    start_time = timeit.default_timer()

    for sample in samples:
        num_docs = len(sample['docs'])

        debug('Cluster: %s (%s)' % (sample['cluster_name'], sample['original_cluster_name']))

        # Prepare docs in cluster
        raw_docs = sample['docs']
        debug('Parsing documents...')
        parsed_docs = parse_docs(raw_docs=raw_docs, lang=LANGUAGE)
        debug('Done.')

        # Clustering sentences into clusters
        debug('Clustering sentences in documents...')
        clusters = clustering_sentences(docs=parsed_docs, cluster_threshold=num_docs // 2, sim_threshold=0.2,
                                        n_top_sentences=None, n_top_clusters=None)
        debug('Done.')

        debug('Ordering clusters...')
        clusters = ordering_clusters(clusters=clusters)
        debug('Done.')

        debug(json.dumps(clusters))

        #     debug('Compressing sentences in each cluster...')
        #     compressed_clusters = compress_clusters(clusters=clusters, num_words=8, max_words=35, num_candidates=200,
        #                                             sim_threshold=0.8, simple_method=True)
        #     debug('Done.')
        #
        #     # Store for other purposes
        #     backup.append(compressed_clusters)
        #
        #     debug('Computing linguistic score...')
        #     scored_clusters = eval_linguistic(clusters=compressed_clusters)
        #     debug('Done.')
        #
        #     debug('Computing informativeness score...')
        #     scored_clusters = eval_informativeness(clusters=scored_clusters)
        #     debug('Done.')
        #
        #     debug('Solving ILP...')
        #     final_sentences = solve_ilp(clusters=scored_clusters, num_words=120, sim_threshold=0.5, greedy_mode=False)
        #     debug('Done.')
        #
        #     debug(normalize_word_suffix(' '.join(final_sentences), lang=LANGUAGE))
        #     write_file(normalize_word_suffix(' '.join(final_sentences), lang=LANGUAGE), sample['save'])
        #
        # debug('Total time: %d s' % (timeit.default_timer() - start_time))
        #
        # # Backup
        # write_json(backup, '../Temp/datasets/%s/backup-%s.json' % (LANGUAGE, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))


def test():
    mapping = {
        '-lcb-': '{',
        '-rcb-': '}',
        '-lrb-': '(',
        '-rrb-': ')'
    }
    clusters_path = '/home/dnanhkhoa/Desktop/Clusters_COSINE_2004_10'
    save_path = '/home/dnanhkhoa/Desktop/result'
    dir_names, dir_paths = read_dir(clusters_path, file_filter=True)

    # dir_paths = dir_paths[0:1]
    idx = 0

    for i, dir_path in enumerate(dir_paths):
        debug(i + idx + 1)
        clusters = []
        file_names, file_paths = read_dir(dir_path, dir_filter=True)
        for file_path in file_paths:
            cluster = []
            cluster_content = read_lines(file_path)
            for sentence in cluster_content:
                tokens = []
                tags = []

                token_list = sentence.strip().split(' ')
                for item in token_list:
                    m = regex.match("^(.+)/(.+)$", item)
                    token, tag = m.group(1), m.group(2)

                    token = token.lower()
                    if token in mapping:
                        token = mapping[token]

                    if is_punctuation(token):
                        tag = 'PUNCT'

                    tokens.append(token)
                    tags.append(tag)

                cluster.append({
                    'tags': tags,
                    'tokens': tokens
                })
            clusters.append(cluster)

        debug('Compressing sentences in each cluster...')
        compressed_clusters = compress_clusters(clusters=clusters, num_words=8, num_candidates=200,
                                                sim_threshold=0.8)
        debug('Done.')

        debug('Computing linguistic score...')
        scored_clusters = pool_executor(fn=eval_linguistic_score, args=compressed_clusters)
        debug('Done.')

        debug('Computing informativeness score...')
        scored_clusters = pool_executor(fn=eval_informativeness_score, args=scored_clusters)
        debug('Done.')

        print(json.dumps(scored_clusters))

        debug('Solving ILP...')
        final_sentences = solve_ilp(clusters=scored_clusters, num_words=120, sim_threshold=0.5)
        debug('Done.')

        debug(normalize_word_suffix(' '.join(final_sentences), lang=LANGUAGE))
        write_file(' '.join(final_sentences), '%s/%d' % (save_path, idx + i + 1))


def test2():
    fnames = ['d30001t', 'd30002t', 'd30003t', 'd30005t', 'd30006t', 'd30007t', 'd30008t', 'd30010t', 'd30011t',
              'd30015t', 'd30017t', 'd30020t', 'd30022t', 'd30024t', 'd30026t', 'd30027t', 'd30028t', 'd30029t',
              'd30031t', 'd30033t', 'd30034t', 'd30036t', 'd30037t', 'd30038t', 'd30040t', 'd30042t', 'd30044t',
              'd30045t', 'd30046t', 'd30047t', 'd30048t', 'd30049t', 'd30050t', 'd30051t', 'd30053t', 'd30055t',
              'd30056t', 'd30059t', 'd31001t', 'd31008t', 'd31009t', 'd31013t', 'd31022t', 'd31026t', 'd31031t',
              'd31032t', 'd31033t', 'd31038t', 'd31043t', 'd31050t']

    clusters_path = '/home/dnanhkhoa/Desktop/clusters'
    save_path = '/home/dnanhkhoa/Desktop/result'
    file_names, file_paths = read_dir(clusters_path, dir_filter=True)

    idx = 0

    for i, file_path in enumerate(file_paths):
        clusters_content = read_json(file_path)

        clusters = []

        for clusterx in clusters_content:
            cluster = []
            for sentence in clusterx:
                parsed = parse(sentence)
                tags = ['PUNCT' if is_punctuation(x) else x for x in parsed[0]['tags']]
                tokens = [x.lower() for x in parsed[0]['tokens']]

                cluster.append({
                    'tags': tags,
                    'tokens': tokens
                })

            clusters.append(cluster)

        debug('Compressing sentences in each cluster...')
        compressed_clusters = compress_clusters(clusters=clusters, num_words=8, num_candidates=200,
                                                sim_threshold=0.8)
        debug('Done.')

        debug('Computing linguistic score...')
        scored_clusters = eval_linguistic_score_clusters(compressed_clusters)
        debug('Done.')

        debug('Computing informativeness score...')
        scored_clusters = eval_informativeness_score_clusters(scored_clusters)
        debug('Done.')

        debug('Solving ILP...')
        final_sentences = solve_ilp(clusters=scored_clusters, num_words=120, sim_threshold=0.5)
        debug('Done.')

        debug(normalize_word_suffix(' '.join(final_sentences), lang=LANGUAGE))
        write_file(' '.join(final_sentences), '%s/%s' % (save_path, fnames[idx + i]))


# OK
def prepare_datasets():
    debug('Preprocessing DUC 2004 dataset...')
    preprocess_duc04('../Datasets/Datasets/DUC04', '../Temp/datasets/en')
    debug('Done.')

    debug('Preprocessing Vietnamese MDS dataset...')
    preprocess_vimds('../Datasets/Datasets/VietnameseMDS', '../Temp/datasets/vi')
    debug('Done.')

    # debug('Preprocessing Vietnamese MDS - HCMUS dataset...')
    # preprocess_vimds_hcmus('../Datasets/Datasets/VietnameseMDS-HCMUS', '../Temp/datasets/vi-hcmus')
    # debug('Done.')


# OK
if __name__ == '__main__':
    if PREPARE_DATASET:
        prepare_datasets()
    else:
        main()
