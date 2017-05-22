#!/bin/bash

curdir=`pwd`
rerankdir=$curdir/../Temp/rerank
rmn=$curdir/../Tools/rmn

cd $rmn

th RM.lua -max_seq_length 80 -min_seq_length 10 -data_dir data/en -num_layers 1 -mem_size 15 -batch_size 1 -emb_size 128 -rnn_size 128 -nhop 1 -gpuid 0 -init_from data/en/model.t7 -rerank_dir $rerankdir -rerank_min_seq_length 5

cd $curdir
