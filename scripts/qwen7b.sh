CUDA_VISIBLE_DEVICES=3 python ../train.py \
    --model_name '/root/huggingface/qwen-2.5-7B-instruct' \
    --dataset_path '/root/LuxunGPT/datasets/qwen-2.5-7B-instruct' \
    --output_dir '/root/LuxunGPT/models/qwen-2.5-7B-instruct-10000' \
    --train_name 'qwen-2.5-7B-instruct-10000' \
    --device 'auto' \

CUDA_VISIBLE_DEVICES=3 python ../inference.py \
    --model_path '/root/huggingface/qwen-2.5-7B-instruct' \
    --peft_model_path '/root/LuxunGPT/models/qwen-2.5-7B-instruct-10000' \
    --max_length 100 \
    --device 'auto' \