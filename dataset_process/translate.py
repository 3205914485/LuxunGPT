import json
from openai import OpenAI
from tqdm import tqdm  # 用于显示进度条
from concurrent.futures import ThreadPoolExecutor, as_completed  # 用于并行处理

def load_json(path):
    """读取json文件"""
    with open(path, 'r', encoding='utf-8') as fp:
        data_list = json.load(fp)
    return data_list

# 加载句子列表
luxun_sts = load_json('sentences.json') # 读取所有句子

# 初始化 OpenAI 客户端
client = OpenAI(api_key="API_KEY", base_url="https://api.deepseek.com")

# 调用 DeepSeek API 翻译句子
def translate_to_vernacular(sentence):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个中文翻译助手，负责将鲁迅写出的文言文或复杂的中文句子翻译为现代白话文。注意保证内容的一致性，信达雅。灵活处理一些古文中的含义，尽量使用更现代化的语句。如果包含一些外文，照常翻译为外语。"},
                {"role": "user", "content": f"将以下句子翻译为白话文，只返回翻译结果：{sentence}"},
            ],
            stream=False
        )
        # 提取翻译结果
        translated_text = response.choices[0].message.content.strip()
        return translated_text if translated_text else sentence  # 返回翻译后的文本，如果为空则返回原句
    except Exception as e:
        print(f"翻译失败: {e}")
        return sentence  # 如果翻译失败，返回原句

# 翻译一个批次的句子并保存到文件
def translate_batch(batch, batch_id, output_dir='translated_batches'):
    translated_batch = [translate_to_vernacular(sentence) for sentence in batch]  # 翻译当前批次
    # 保存当前批次的翻译结果
    output_file = f"{output_dir}/batch_{batch_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_batch, f, ensure_ascii=False, indent=4)
    return output_file

# 并行翻译所有句子
def translate_parallel(sentences, batch_size=20, output_dir='translated_batches'):
    import os
    os.makedirs(output_dir, exist_ok=True)  # 创建保存批次的目录

    # 将句子列表分成多个批次
    batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

    # 使用线程池并行翻译
    with ThreadPoolExecutor(max_workers=10) as executor:  # 设置最大线程数
        futures = []
        for batch_id, batch in enumerate(batches):
            futures.append(executor.submit(translate_batch, batch, batch_id, output_dir))

        # 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="翻译进度"):
            pass  # 等待所有任务完成

    print("所有批次翻译完成。")

# 合并所有批次的翻译结果
def merge_results(output_dir='translated_batches', final_output_file='translated_sentences.json'):
    import os
    import re

    translated_sentences = []
    # 获取所有批次文件
    batch_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    # 按照文件名中的数字排序
    batch_files.sort(key=lambda x: int(re.search(r'batch_(\d+)\.json', x).group(1)))

    # 按顺序读取并合并文件
    for file_name in batch_files:
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            translated_sentences.extend(json.load(f))

    # 保存合并后的结果
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_sentences, f, ensure_ascii=False, indent=4)
    print(f"所有翻译结果已合并到 {final_output_file}。")

# 开始翻译
# translate_parallel(luxun_sts)
merge_results()