#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out


model_ids=("llama13b-SRIBD-pretrained-raw-check0-227566" "jais-13b")

batch_size=1
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


