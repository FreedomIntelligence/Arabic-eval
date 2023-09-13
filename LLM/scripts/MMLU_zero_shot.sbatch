#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

model_id=phoenix-inst-chat-7b

batch_size=1


accelerate launch \
    eval.py \
    --model_id ${model_id} \
    --batch_size ${batch_size} \
    --benchmark_name MMLU \
    --generation_type MMLU \
    --setting zero_shot
    
