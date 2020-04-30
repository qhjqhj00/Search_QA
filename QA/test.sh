#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

export MODEL_NAME=robert
export OUTPUT_DIR=../output
export BERT_DIR=../model/robert
export GLUE_DIR=../data # set your data dir
TASK_NAME="CMRC2018"

python test_mrc.py \
  --gpu_ids="0,1,2" \
  --n_batch=192 \
  --max_seq_length=512 \
  --max_ans_length=300 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_restore_dir=$OUTPUT_DIR/$MODEL_NAME/ \
  --output_dir=$OUTPUT_DIR/$MODEL_NAME/ \
  --test_dir1=$GLUE_DIR/test_examples.json \
  --test_dir2=$GLUE_DIR/test_features.json \
  --test_file=$GLUE_DIR/squad_test.json \




