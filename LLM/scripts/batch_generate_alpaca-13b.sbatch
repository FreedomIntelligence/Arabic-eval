#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_ids=("llama-ace-v5.0" "llama-pretrained-v5.2" "bloomz-ace-v5.0" "phoenix-inst-chat-7b" "phoenix-multiple-langs-v1")
model_ids=("llama-pretrained-13b")
#  "llama-pretrained-13b"
# "jais-chat-13b"
for i in "${!model_ids[@]}"; do
    accelerate launch \
        generate.py \
        --model_id ${model_ids[i]} \
        --generation_type greedy \
        --data_name alpaca_eval_ar \
        --batch_size 1 \
        --device_placement
done


