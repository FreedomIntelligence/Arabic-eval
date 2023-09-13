#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_ids=("llama13b-SRIBD-pretrained-raw-check0-227566" "jais-13b")

batch_size=1
n_shot=5
for i in "${!model_ids[@]}"; do
    accelerate launch \
        eval.py \
        --model_id ${model_ids[i]} \
        --batch_size ${batch_size} \
        --benchmark_name ArabicCulture \
        --setting few_shot \
        --n_shot ${n_shot} \
        --generation_type ArabicCulture
done

