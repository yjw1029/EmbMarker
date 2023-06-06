#!/bin/bash

cd src
accelerate launch run_gpt_backdoor.py \
--seed 2022 \
--model_name_or_path bert-base-cased \
--per_device_train_batch_size 32 \
--max_length 128 \
--selected_trigger_num 20 \
--max_trigger_num 4 \
--trigger_min_max_freq 0.005 0.01 \
--output_dir ../output \
--gpt_emb_train_file ../data/emb_enron_train \
--gpt_emb_validation_file ../data/emb_enron_test \
--gpt_emb_test_file ../data/emb_enron_test \
--cls_learning_rate 1e-2 \
--cls_num_train_epochs 3 \
--cls_hidden_dim 256 \
--cls_dropout_rate 0.2 \
--copy_learning_rate 5e-5 \
--copy_num_train_epochs 3 \
--transform_hidden_size 1536 \
--transform_dropout_rate 0.0 \
--with_tracking \
--report_to wandb \
--job_name enron_adv \
--word_count_file ../data/word_countall.json \
--data_name enron \
--project_name embmarker \
--use_copy_target True