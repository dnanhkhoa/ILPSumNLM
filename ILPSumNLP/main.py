#!/usr/bin/python
# -*- coding: utf8 -*-
import spacy
from utils import *
# Configs


def load_models():
    debug('Load models...')
    spacy.util.set_data_path('data/spacy')
    model = spacy.load('en_core_web_md-1.2.1')
    debug('Done')


def main():
    load_models()


if __name__ == '__main__':
    main()
