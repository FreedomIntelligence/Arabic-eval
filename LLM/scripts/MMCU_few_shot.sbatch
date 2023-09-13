#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

model_id=phoenix-inst-chat-7b
n_shot=5

batch_size=1


accelerate launch \
    eval.py \
    --model_id ${model_id} \
    --batch_size ${batch_size} \
    --benchmark_name MMCU \
    --generation_type MMCU \
    --setting few_shot \
    --n_shot ${n_shot}
    
