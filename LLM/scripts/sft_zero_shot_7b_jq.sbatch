#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

# model_ids=("llama-raw-pretrained-sencondround-ckpt0-430000" "llama-raw-pretrained-ckpt0-227566" "llama-2-7b-hf" "bloomz-7b1-mt")
# export CUDA_VISIBLE_DEVICES=5
# model_ids=("bloomz-ace-v5.0" )
# batch_size=1
# for i in "${!model_ids[@]}"; do
#     accelerate launch \
#         eval.py \
#         --model_id ${model_ids[i]} \
#         --batch_size ${batch_size} \
#         --benchmark_name EXAMS_Arabic \
#         --setting zero_shot \
#         --generation_type EXAMS
# done

# export CUDA_VISIBLE_DEVICES=6
# model_ids=( "llama-pretrained-v5.2")
# batch_size=1
# for i in "${!model_ids[@]}"; do
#     accelerate launch \
#         eval.py \
#         --model_id ${model_ids[i]} \
#         --batch_size ${batch_size} \
#         --benchmark_name EXAMS_Arabic \
#         --setting zero_shot \
#         --generation_type EXAMS
# done


# export CUDA_VISIBLE_DEVICES=0
# model_ids=("llama-raw-pretrained-sencondround-ckpt0-430000")
# batch_size=2
# n_shot=5
# for i in "${!model_ids[@]}"; do
#     accelerate launch \
#         eval.py \
#         --model_id ${model_ids[i]} \
#         --batch_size ${batch_size} \
#         --benchmark_name ArabicCulture \
#         --setting few_shot \
#         --n_shot ${n_shot} \
#         --generation_type ArabicCulture
# done


# export CUDA_VISIBLE_DEVICES=1
# model_ids=("llama-raw-pretrained-sencondround-ckpt0-430000")
# batch_size=1
# n_shot=5
# for i in "${!model_ids[@]}"; do
#     accelerate launch \
#         eval.py \
#         --model_id ${model_ids[i]} \
#         --batch_size ${batch_size} \
#         --benchmark_name ArabicCulture \
#         --setting few_shot \
#         --n_shot ${n_shot} \
#         --generation_type ArabicCulture
# done