#!/usr/bin/python
# -*- coding: utf8 -*-
import json
import os

import gensim
import requests

DEBUG = True

APP_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def full_path(*rel_path):
    return os.path.join(APP_PATH, *rel_path)


def path_info(path):
    path = full_path(path)
    if os.path.exists(path):
        if os.path.isfile(path):
            return 1
        if os.path.isdir(path):
            return 0
    return -1


def make_dirs(dir_name):
    dir_name = full_path(dir_name)
    if path_info(dir_name) != 0:
        os.makedirs(dir_name)


def read_lines(file_name):
    file_name = full_path(file_name)
    assert path_info(file_name) == 1, 'File does not exist!'
    lines = []
    with open(file_name, 'rb') as f:
        try:
            for line in f:
                lines.append(line.decode('UTF-8').rstrip('\r\n'))
        except Exception as e:
            debug(e)
    return lines


def write_lines(lines, file_name, end_line='\n'):
    file_name = full_path(file_name)
    make_dirs(os.path.dirname(file_name))
    with open(file_name, 'wb') as f:
        try:
            for line in lines:
                f.write((line + end_line).encode('UTF-8'))
        except Exception as e:
            debug(e)


def read_file(file_name):
    file_name = full_path(file_name)
    assert path_info(file_name) == 1, 'File does not exist!'
    with open(file_name, 'rb') as f:
        try:
            return f.read().decode('UTF-8')
        except Exception as e:
            debug(e)


def write_file(data, file_name):
    file_name = full_path(file_name)
    make_dirs(os.path.dirname(file_name))
    with open(file_name, 'wb') as f:
        try:
            f.write(data.encode('UTF-8'))
        except Exception as e:
            debug(e)


def read_json(file_name):
    file_name = full_path(file_name)
    assert path_info(file_name) == 1, 'File does not exist!'
    with open(file_name, 'rb') as f:
        try:
            return json.loads(f.read().decode('UTF-8'))
        except Exception as e:
            debug(e)


def write_json(obj, file_name):
    file_name = full_path(file_name)
    make_dirs(os.path.dirname(file_name))
    with open(file_name, 'wb') as f:
        try:
            f.write(json.dumps(obj, indent=4, ensure_ascii=False).encode('UTF-8'))
        except Exception as e:
            debug(e)


def parse(docs, lang='en'):
    try:
        server_url = '0.0.0.0'
        server_port = 5100 if lang is 'en' else 5105
        data = {
            'text': [doc.encode('UTF-8') for doc in docs] if isinstance(docs, list) else docs.encode('UTF-8')
        }
        response = requests.post(url='http://%s:%d/handle' % (server_url, server_port), data=data)
        if response.status_code == 200:
            result = json.loads(response.content.decode('UTF-8'))
            if not result.get('status', False):
                raise Exception(result.get('error', 'Undefined error'))
            return result.get('result', None)
    except Exception as e:
        debug(e)
    return None


def read_dir(dir_path, dir_filter=False, file_filter=False, ext_filter=None):
    full_dir_path = full_path(dir_path)
    assert path_info(full_dir_path) == 0, 'Folder does not exist!'

    paths = []
    files = []

    try:
        sorted_files = sorted(os.listdir(full_dir_path))
        for file in sorted_files:
            file_path = '%s/%s' % (dir_path, file)
            full_file_path = full_path(file_path)
            file_info = path_info(full_file_path)
            if file_info == 1:  # File
                parts = os.path.splitext(file)
                if not file_filter and (ext_filter is None or parts[1] not in ext_filter):
                    files.append(file)
                    paths.append(file_path)
            elif file_info == 0:  # Folder
                if not dir_filter:
                    files.append(file)
                    paths.append(file_path)

    except Exception as e:
        debug(e)

    return files, paths


def build_doc2vec_model(train_file, model_file):
    # Doc2vec parameters
    vector_size = 300
    window_size = 15
    min_count = 1
    sampling_threshold = 1e-5
    worker_count = 20  # Number of parallel processes
    hs = 0
    dm = 0  # 0 = dbow; 1 = dmpv
    negative_size = 5
    dbow_words = 1
    dm_concat = 1
    num_epoch = 100

    docs = gensim.models.doc2vec.TaggedLineDocument(train_file)
    model = gensim.models.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count,
                                  sample=sampling_threshold, workers=worker_count, hs=hs, dm=dm, negative=negative_size,
                                  dbow_words=dbow_words, dm_concat=dm_concat, iter=num_epoch)

    # Save model
    model.save(model_file)
