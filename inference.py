import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def load_model_and_tokenizer(model_path, peft_model_path, device):
    # 加载基础模型和分词器
    # q_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=args.use_8bit)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device, load_in_8bit=args.use_8bit)
    
    # 加载 LoRA 微调后的模型
    if args.mode == 'lora':
        print('lora mode')
        model = PeftModel.from_pretrained(model, peft_model_path)

    model.eval()
    return model, tokenizer

def inference(model, tokenizer, prompt, max_length=100):
    # 对输入进行编码
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    # 生成输出
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_file(input_file, output_file, model, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile):
            prompt = line.strip()
            if prompt:
                response = inference(model, tokenizer, prompt, max_length=args.max_length)
                outfile.write(f"{response}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for LuxunGPT model.")
    
    # 模型和分词器路径
    parser.add_argument("--model_path", type=str, default="../../huggingface/chatglm3-6b", help="Path to the base model")
    parser.add_argument("--peft_model_path", type=str, default="./models/chatglm-10000", help="Path to the PEFT model")
    
    # 输入文件和输出文件路径
    parser.add_argument("--input_file", type=str, default="/root/LuxunGPT/datasets/rewritten_prompts.txt", help="Path to the input file with prompts")
    
    # 其他参数
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the generated output")
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    parser.add_argument('--mode', type=str, default='lora', help='which mode for inference') 
    parser.add_argument("--use_8bit", type=bool, default=False, help="是否8bit量化") 

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.peft_model_path, args.device)

    if args.mode == 'lora':
        print('lora mode')
        output_file = f'/root/LuxunGPT/outputs/{args.peft_model_path.split("/")[-1]}.txt'
    else:
        output_file = f'/root/LuxunGPT/outputs/{args.model_path.split("/")[-1]}_org.txt' 
    # 处理文件
    process_file(args.input_file, output_file, model, tokenizer)