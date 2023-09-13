#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out


model_ids=("llama-raw-pretrained-sencondround-ckpt0-430000" "llama-raw-pretrained-ckpt0-227566" "llama-2-7b-hf" "bloomz-7b1-mt")


batch_size=8
n_shot=5
for i in "${!model_ids[@]}"; do
    accelerate launch \
        eval.py \
        --model_id ${model_ids[i]} \
        --batch_size ${batch_size} \
        --benchmark_name EXAMS_Arabic \
        --setting few_shot \
        --n_shot ${n_shot} \
        --generation_type EXAMS
done


