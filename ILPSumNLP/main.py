#!/usr/bin/python
# -*- coding: utf8 -*-

import kenlm

from utils.utils import *

kenlm_model = kenlm.Model('')

stopwords_vi = read_lines('')
stopwords_en = read_lines('')

'''
    size = len(s.split(' '))
    score = model.score(s)
    score = (score / (size + 1))
    score = 1. / (1 - score)
    print(score)
'''


# preprocess_duc04('../Datasets/DUC04', '../Temp/datasets/en')
# preprocess_vimds('../Datasets/VietnameseMDS', '../Temp/datasets/vi')


def main():
    pass


if __name__ == '__main__':
    main()
