#!/usr/bin/python
# -*- coding: utf8 -*-
import regex

from utils.normalizer import *
from utils.utils import *


def preprocess_duc04(dir_path, save_path):
    num_docs = 500
    num_refs = 200

    docs_count = 0
    refs_count = 0

    content_pattern = regex.compile(r'<TEXT>(.+?)<\/TEXT>', regex.DOTALL | regex.IGNORECASE)

    clusters_dir_path = '%s/clusters' % save_path
    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(peers_dir_path)

    ref_docs, refs_path = read_dir('%s/models' % dir_path, dir_filter=True)

    clusters, clusters_path = read_dir('%s/docs' % dir_path, file_filter=True)

    for i, cluster in enumerate(clusters):
        # Docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)
        for j, doc in enumerate(docs):
            file_name = '%s/cluster_%d/%d' % (clusters_dir_path, i + 1, j + 1)
            file_content = read_file(docs_path[j])
            matcher = content_pattern.search(file_content)
            if matcher is not None:
                file_content = matcher.group(1).strip()
                # Preprocessing
                write_file(normalize_document(file_content), file_name)

                docs_count += 1

        # Refs
        ref_id = 0
        for j, ref_doc in enumerate(ref_docs):
            if cluster.lower()[:-1] in ref_doc.lower():
                file_name = '%s/cluster_%d/%d' % (models_dir_path, i + 1, ref_id + 1)
                file_content = read_file(refs_path[j])
                # Preprocessing
                write_file(normalize_document(file_content), file_name)
                ref_id += 1
                refs_count += 1

    assert docs_count == num_docs, 'There should be the same number of docs in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'


def preprocess_vimds(dir_path, save_path):
    docs_dir_path = '%s/docs' % save_path
    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(peers_dir_path)
