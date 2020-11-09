#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=robert
export OUTPUT_DIR=../output
export BERT_DIR=../model/robert
export GLUE_DIR=../data # set your data dir
TASK_NAME="CMRC2018"

python run_mrc.py \
  --gpu_ids="1,2,3,4" \
  --train_epochs=2 \
  --n_batch=16 \
  --lr=3e-5 \
  --warmup_rate=0.1 \
  --max_seq_length=512 \
  --max_ans_length=300\
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_restore_dir=$BERT_DIR/pytorch_model.pth \
  --train_dir=$GLUE_DIR/train_features.json \
  --train_file=$GLUE_DIR/train_squad.json \
  --dev_dir1=$GLUE_DIR/dev_examples.json \
  --dev_dir2=$GLUE_DIR/dev_features.json \
  --dev_file=$GLUE_DIR/dev_squad.json \
  --checkpoint_dir=$OUTPUT_DIR/$MODEL_NAME/





