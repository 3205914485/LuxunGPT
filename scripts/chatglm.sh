CUDA_VISIBLE_DEVICES=3 python ../train.py \
    --model_name '/root/huggingface/chatglm3-6b' \
    --dataset_path '/root/LuxunGPT/datasets/chatglm3-6b' \
    --output_dir '/root/LuxunGPT/models/chatglm3-6b-10000' \
    --train_name 'chatglm3-6b-10000' \
    --device 'cuda:0' \

CUDA_VISIBLE_DEVICES=3 python ../inference.py \
    --model_path '/root/LuxunGPT/datasets/chatglm3-6b' \
    --peft_model_path '/root/LuxunGPT/models/chatglm3-6b-10000' \
    --max_length 100 \
    --device 'cuda:0' \