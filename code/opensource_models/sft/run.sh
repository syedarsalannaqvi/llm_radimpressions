#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5


export TRANSFORMERS_CACHE=/PATH/TO/CACHE/.cache

python run_causal_models.py \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --train_file "/home/ssaeidi1/sft/data/df_train.csv" \
        --validation_file "/home/ssaeidi1/sft/data/df_val.csv" \
        --test_file "/home/ssaeidi1/sft/data/df_test.csv" \
        --output_dir /data/data/amir/mayo/mistral_7b \
        --per_device_train_batch_size="4" \
        --per_device_eval_batch_size="4" \
        --gradient_accumulation_steps="2" \
        --predict_with_generate \
        --num_train_epochs="3" \
        --save_strategy=no
