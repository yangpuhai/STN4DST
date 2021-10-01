#!/usr/bin/env bash

# dataset can be "DSTC2", "WOZ2.0", "sim-M", or "sim-R", "MultiWOZ2.2"
dataset="DSTC2"
# Directory of the pre-trained [BERT-Base, Uncased] model
PRETRAINED_BERT=bert-base-uncased

nohup python -u evaluation.py \
  --dataset=${dataset} \
  --n_history=0 \
  --max_seq_length=150 \
  --vocab_path=${PRETRAINED_BERT}/vocab.txt \
  --bert_config_path=${PRETRAINED_BERT}/bert_config_base_uncased.json \
  > ${dataset}_test.log 2>&1 &