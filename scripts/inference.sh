#!/bin/bash

max_length=200
models=('qwen-2.5-7B-instruct')

for model in "${models[@]}"; do
    # 运行 LoRA 模式
    python ../inference.py \
        --model_path "/root/huggingface/${model}" \
        --peft_model_path "/root/LuxunGPT/models/${model}-10000" \
        --max_length $max_length \
        --device 'auto' \
        --mode 'lora' \
        # --use_8bit 1 \

    # 运行原始模式
    python ../inference.py \
        --model_path "/root/huggingface/${model}" \
        --peft_model_path "/root/LuxunGPT/models/${model}-10000" \
        --max_length $max_length \
        --device 'auto' \
        --mode 'org'
done