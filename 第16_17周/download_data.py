import os
from datasets import load_dataset
import json

# 1. 设置 HF 镜像（国内加速）
# 这一步非常重要，否则在国内下载 Hugging Face 数据集可能会非常慢或超时
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_alpaca_data():
    print("🚀 开始下载 alpaca_chinese 数据集...")
    
    try:
        # 加载数据集
        # path: 数据集名称
        # split="train": 只加载训练集
        # 替换为更稳定的数据集源：silk-road/alpaca-data-gpt4-chinese
        dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
        
        print(f"✅ 下载完成！数据集包含 {len(dataset)} 条数据。")
        print("🔍 数据预览（前1条）：")
        print(dataset[0])
        
        # 保存为本地 JSON 文件，方便后续查看和处理
        # force_ascii=False: 保证中文正常显示，不是 \uXXXX
        # 使用绝对路径，确保文件生成在脚本同级目录
        output_file = os.path.join(os.path.dirname(__file__), "raw_data.json")
        
        # 修复：直接保存为标准的 JSON 列表格式，而不是奇怪的拼接格式
        # dataset.to_json 默认是一行一个 JSON (JSONL)，但参数是 indent=4 导致它输出了带缩进的 JSONL，非常诡异
        # 我们手动转换成 Python 列表再保存，最稳妥
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(list(dataset), f, ensure_ascii=False, indent=4)
            
        print(f"💾 原始数据已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 建议检查网络或确认 HF_ENDPOINT 是否生效")

if __name__ == "__main__":
    download_alpaca_data()
