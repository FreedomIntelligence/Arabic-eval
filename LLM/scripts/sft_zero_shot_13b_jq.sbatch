#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out



# export CUDA_VISIBLE_DEVICES=5
# datasets=("ArabicCulture" "EXAMS_Arabic" "MMLUArabic")
# types=("ArabicCulture" "EXAMS" "MMLU")
# for i in "${!datasets[@]}"; do
#     accelerate launch \
#         eval.py \
#         --model_id jais-13b\
#         --batch_size 1 \
#         --benchmark_name ${datasets[i]} \
#         --setting zero_shot \
#         --generation_type ${types[i]}
# done
#第一段代码块作为后台任务执行
{
    export CUDA_VISIBLE_DEVICES=5
    datasets=("ArabicCulture" "EXAMS_Arabic" "MMLUArabic")
    types=("ArabicCulture" "EXAMS" "MMLU")
    for i in "${!datasets[@]}"; do
        accelerate launch \
            eval.py \
            --model_id jais-13b\
            --batch_size 1 \
            --benchmark_name ${datasets[i]} \
            --setting few_shot \
            --n_shot 5 \
            --generation_type ${types[i]}
    done 
}  &  # 后台任务

# # 第二段代码块作为后台任务执行
{
    export CUDA_VISIBLE_DEVICES=6
    datasets=("ArabicCulture" "EXAMS_Arabic" "MMLUArabic")
    types=("ArabicCulture" "EXAMS" "MMLU")
    for i in "${!datasets[@]}"; do
        accelerate launch \
            eval.py \
            --model_id llama13b-SRIBD-pretrained-raw-check0-227566\
            --batch_size 1 \
            --benchmark_name ${datasets[i]} \
            --setting few_shot \
            --n_shot 5 \
            --generation_type ${types[i]}
    done
} &  # 后台任务

{
    export CUDA_VISIBLE_DEVICES=7
    datasets=("ArabicCulture" "EXAMS_Arabic" "MMLUArabic")
    types=("ArabicCulture" "EXAMS" "MMLU")
    for i in "${!datasets[@]}"; do
        accelerate launch \
            eval.py \
            --model_id llama-pretrained-13b\
            --batch_size 1 \
            --benchmark_name ${datasets[i]} \
            --setting zero_shot \
            --generation_type ${types[i]}
    done
} &

# # 等待后台任务完成
wait

echo "所有代码块执行完毕"
