#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

export MODEL_NAME=robert
export OUTPUT_DIR=../output
export BERT_DIR=robert
export GLUE_DIR=data # set your data dir
TASK_NAME="CMRC2018"

python agg_test_mrc.py \
  --gpu_ids="1" \
  --n_batch=384 \
  --max_seq_length=128 \
  --max_ans_length=500 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --output_dir=$OUTPUT_DIR/ \
  --test_dir1=$GLUE_DIR/test_examples.json \
  --test_dir2=$GLUE_DIR/test_features.json \
  --test_file=$GLUE_DIR/test_toy.json \
  --agg_model_dir=../agg_model/
  


