#!/usr/bin/python
# -*- coding: utf8 -*-
from utils import *


def main():
    # peers_1 = '../Temp/datasets/vi/peers/'
    # peers_2 = '../Temp/datasets/vi/peers_/'
    # score_1 = '../Temp/datasets/vi/rouge_result.out'
    # score_2 = '../Temp/datasets/vi/rouge_result.out_'
    #
    # sc_1 = read_lines(score_1)
    # sc_2 = read_lines(score_2)
    #
    # nsc_1 = []
    # for l1 in sc_1:
    #     p1 = l1.split(' ')
    #     if len(p1) <= 1 or p1[1] != 'ROUGE-2' or p1[2] != 'Eval':
    #         continue
    #     nsc_1.append((float(p1[4][2:]), p1[3][:-7]))
    #
    # nsc_2 = []
    # for l2 in sc_2:
    #     p2 = l2.split(' ')
    #     if len(p2) <= 1 or p2[1] != 'ROUGE-2' or p2[2] != 'Eval':
    #         continue
    #     nsc_2.append((float(p2[4][2:]), p2[3][:-7]))
    #
    # for i in range(len(nsc_1)):
    #     if nsc_1[i][0] < nsc_2[i][0]:
    #         shutil.copy(src=full_path(peers_2 + nsc_2[i][1]), dst=full_path(peers_1 + nsc_1[i][1]))
    l = 0.
    c = 0
    with open('/home/dnanhkhoa/Desktop/data/europarl-v7.en', 'r') as f:
        for line in f:
            num_words = get_num_words(line)
            l += num_words
            c += 1
            # p = parse(line)
            # ss = []
            # for x in p:
            #     ss.append(' '.join(x['tokens']))
            # for s in ss:
            #     num_words = get_num_words(s)
            #     l += num_words
            #     c += 1
            #     debug(c)

    debug(l / c)


if __name__ == '__main__':
    main()
