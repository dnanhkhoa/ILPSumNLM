#!/usr/bin/python
# -*- coding: utf8 -*-

import json
import os

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
        server_url = 'http://127.0.0.1'
        server_port = 5100 if lang is 'en' else 5105
        data = {'text': [doc.encode('UTF-8') for doc in docs] if isinstance(docs, list) else docs.encode('UTF-8')}
        response = requests.post(url='%s:%d/handle' % (server_url, server_port), data=data)
        if response.status_code == 200:
            result = json.loads(response.content.decode('UTF-8'))
            if not result.get('status', False):
                raise Exception(result.get('error', 'Undefined error'))
            return result.get('result', None)
    except Exception as e:
        debug(e)
    return None
