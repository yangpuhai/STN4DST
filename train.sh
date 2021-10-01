#!/usr/bin/env bash
# platform
# OS: Red Hat 4.8.5-11 (Linux version 3.10.0-514.26.2.el7.x86_64)
# GPU: GeForce GTX TITAN X (memory: 12G)
# CPU: Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz
# CUDA Version: 9.0.176
# NVRM version: NVIDIA UNIX x86_64 Kernel Module  450.66
# GCC version:  gcc version 4.8.5 20150623
# python version: Python 3.6.9
# pytorch version: torch 1.0.0

# dataset can be "DSTC2", "WOZ2.0", "sim-M", or "sim-R", "MultiWOZ2.2"
dataset="MultiWOZ2.2"
n_history=0
random_seed=42
# Directory of the pre-trained [BERT-Base, Uncased] model
PRETRAINED_BERT=bert-base-uncased

nohup python -u train.py \
  --dataset=${dataset} \
  --batch_size=16 \
  --lr=4e-5 \
  --n_epochs=100 \
  --patience=15 \
  --dropout=0.1 \
  --word_dropout=0.1 \
  --value_dropout=0.0 \
  --random_seed=${random_seed} \
  --n_history=${n_history} \
  --max_seq_length=220 \
  --vocab_path=${PRETRAINED_BERT}/vocab.txt \
  --bert_config_path=${PRETRAINED_BERT}/bert_config_base_uncased.json \
  --bert_ckpt_path=${PRETRAINED_BERT}/pytorch_model.bin \
  > ${dataset}_seed[${random_seed}]_history[${n_history}]_train.log 2>&1 &