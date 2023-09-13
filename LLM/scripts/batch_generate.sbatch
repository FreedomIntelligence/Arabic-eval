#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out


model_ids=("llama-ace-v5.0" "llama-pretrained-v5.2" "bloomz-ace-v5.0" "llama-pretrained-13b")
temperature_list=("0.2" "0.4" "0.6" "0.8" "1")

for i in "${!model_ids[@]}"; do
    mv generation/outputs/vicuna_ar80/${model_ids[i]}.jsonl generation/outputs/vicuna_ar80/${model_ids[i]}_greedy.jsonl
    for j in "${!generation_types[@]}"; do
        accelerate launch \
            generate.py \
            --model_id ${model_ids[i]} \
            --generation_type sample${temperature_list[j]} \
            --data_name vicuna_ar80 \
            --batch_size 4 \
            --device_placement
        mv generation/outputs/vicuna_ar80/${model_ids[i]}.jsonl generation/outputs/vicuna_ar80/${model_ids[i]}_t${temperature_list[i]}.jsonl
done


