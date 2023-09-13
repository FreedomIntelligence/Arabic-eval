#!/bin/bash
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output scripts/logs/job-%j.out

model_id=phoneix
checkpoint_dir=/mntcephfs/data/med/zhihong/workspace/chimera/phoenix-inst-chat-7b
system_prompt="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\nHuman: <s>{question}</s>Assistant: <s>"
data_name=vicuna_ar80

batch_size=8


accelerate launch \
    generate_from_model.py \
    --model_id ${model_id} \
    --model_path ${checkpoint_dir} \
    --system_prompt "${system_prompt}" \
    --data_name ${data_name} \
    --do_sample \
    --max_length 2048 \
    --temperature 0.7 \
    --top_k 50 \
    --batch_size ${batch_size} \
    --device_placement

