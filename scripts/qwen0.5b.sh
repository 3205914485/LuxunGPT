# python ../train.py \
#     --model_name '/root/huggingface/qwen-2.5-0.5B-instruct' \
#     --dataset_path '/root/LuxunGPT/datasets/qwen-2.5-0.5B-instruct' \
#     --output_dir '/root/LuxunGPT/models/qwen-2.5-0.5B-instruct-30000' \
#     --train_name 'qwen-2.5-0.5B-instruct-30000' \
#     --device 'auto' \
#     --train_subset_size 30000 \

python ../inference.py \
    --model_path '/root/huggingface/qwen-2.5-0.5B-instruct' \
    --peft_model_path '/root/LuxunGPT/models/qwen-2.5-0.5B-instruct-30000' \
    --max_length 100 \
    --device 'auto' \