from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from datetime import datetime

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename='app_logs.txt',  # Log to file
    level=logging.INFO,       # Log level
    format='%(asctime)s - %(message)s',  # Log format with timestamp
)

# Add a stream handler to log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)

logging.getLogger().addHandler(console_handler)

model_dict = {
    'model1': 'qwen-2.5-0.5B-instruct',
    'model2': 'qwen-2.5-1.5B-instruct',
    'model3': 'qwen-2.5-3B-instruct',
    'model4': 'chatglm3-6b',
    'model5': 'qwen-2.5-7B-instruct',
}

# Dictionary to keep track of loaded models and their memory usage
loaded_models = {}
loaded_mem_history = {
    'model1': 2.3 * 1024**3,
    'model2': 6.3 * 1024**3,
    'model3': 12  * 1024**3,
    'model4': 12  * 1024**3,
    'model5': 0.2 * 1024**3, # put on cpu
}
alpha = 0.95
max_gpu_memory = alpha * 22 * 1024**3  # Set a threshold

# Function to load the model and tokenizer based on the model path
def load_model_and_tokenizer(model_path, peft_model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device)
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()
    return model, tokenizer

# Unload a model from memory
def unload_model(model_key):
    if model_key in loaded_models:
        model = loaded_models[model_key]['model']
        del model
        torch.cuda.empty_cache()  # Clear GPU memory cache
        del loaded_models[model_key]
        logging.info(f"Unloaded model: {model_key}")

# Manage the GPU memory by checking the total memory usage and unloading if necessary
def load_model_to_cpu(model_key):

    model_path = f"/root/huggingface/{model_dict[model_key]}"  
    peft_model_path = f"/root/LuxunGPT/models/{model_dict[model_key]}-10000"  
    # Load the model on CPU first (without moving it to GPU)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cpu")
    model = PeftModel.from_pretrained(model, peft_model_path)
    
    # Estimate memory usage
    if model_key in loaded_mem_history:
        estimated_memory = loaded_mem_history[model_key] #use the gpu-mem directly
    else:
        num_params = sum(p.numel() for p in model.parameters())
        estimated_memory = num_params * 2  # 2 bytes per parameter (float16)
    
    return model, tokenizer, estimated_memory

def get_mem_from_gpu():
    return torch.cuda.memory_allocated()

# Function to manage GPU memory, load model to GPU if enough memory is available
def manage_gpu_memory(model_key, device):

    if model_key in loaded_models:
        return loaded_models[model_key]['model'], loaded_models[model_key]['tokenizer']
    
    # Load model to CPU and estimate memory usage
    model, tokenizer, estimated_memory = load_model_to_cpu(model_key)
    
    # Get current GPU memory usage
    total_memory_used = sum([loaded_models[m]['memory'] for m in loaded_models if loaded_models[m].get('gpu', False)])
    
    # Check if there is enough memory available
    if total_memory_used + estimated_memory > max_gpu_memory:
        # If not enough space, unload the least used GPU model
        while total_memory_used + estimated_memory > max_gpu_memory:
            # Find the least used model currently on the GPU
            least_memory_model_key = min(
                (m for m in loaded_models if loaded_models[m].get('gpu', False)),
                key=lambda m: loaded_models[m]['memory'],
                default=None
            )
            if least_memory_model_key:
                unload_model(least_memory_model_key)
                total_memory_used = sum([loaded_models[m]['memory'] for m in loaded_models if loaded_models[m].get('gpu', False)])
  
    # Now that we have enough memory, move model to GPU
    is_on_gpu = model_key != 'model5'
    if is_on_gpu:
        model = model.to(device)

    true_model_memory = get_mem_from_gpu() - total_memory_used
    # Store the new model and its memory usage
    loaded_models[model_key] = {'model': model, 'tokenizer': tokenizer, 'memory': true_model_memory, 'gpu': is_on_gpu}
    loaded_mem_history[model_key] = true_model_memory
    logging.info(f"Loaded model: {model_key} with {true_model_memory / (1024 ** 2):.2f} MB")

    return model, tokenizer

# Inference function with logging
def inference(model, tokenizer, prompt, max_length=200, model_key=None, device='cuda:0'):

    if model_key == 'model5':
        inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device) 

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log the request and response
    log_request_response(prompt, response, model_key)
    
    return response

# Function to log input and output
def log_request_response(input_text, response_text, model_key):
    log_message = f"Model: {model_key}, Input: {input_text}, Response: {response_text}"
    logging.info(log_message)

# Main page route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Model generation route (Generalized for all models)
@app.route('/generate_<model_key>', methods=["POST"])
def generate(model_key):
    device = "cuda:0"

    # 管理GPU内存 
    model, tokenizer = manage_gpu_memory(model_key, device)

    # 从请求中获取输入文本
    data = request.json
    input_text = data.get("input_text", "")

    if not input_text:
        error_message = "input_text 是必需的参数"
        logging.error(f"错误：{model_key} {error_message}")
        return jsonify({"error": error_message}), 400
    else:
        generated_text = inference(model, tokenizer, input_text, model_key=model_key, device=device)
        return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)