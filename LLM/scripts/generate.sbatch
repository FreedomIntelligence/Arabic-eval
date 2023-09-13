#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

model_id=llama-pretrained-13b
generation_type=greedy
data_name=vicuna_ar80

batch_size=1


accelerate launch \
    generate.py \
    --model_id ${model_id} \
    --generation_type ${generation_type} \
    --data_name ${data_name} \
    --batch_size ${batch_size} \
    --device_placement

