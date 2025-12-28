import json
import pandas as pd
import os

def clean_data():
    # 使用绝对路径，确保文件在脚本同级目录
    base_dir = os.path.dirname(__file__)
    input_file = os.path.join(base_dir, "raw_data.json")
    output_file = os.path.join(base_dir, "processed_data.jsonl")
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件 {input_file}，请先运行 download_data.py")
        return

    print("🧹 开始清洗数据...")
    
    # 1. 读取数据
    # 使用 Pandas 读取 JSON，处理效率更高
    try:
        # 尝试标准 JSON 格式 (list of dicts)
        df = pd.read_json(input_file)
    except ValueError:
        try:
             # 尝试 JSON Lines 格式
            df = pd.read_json(input_file, lines=True)
        except ValueError:
             print("❌ 无法读取数据，请重新运行 download_data.py 生成标准格式的数据。")
             return
    
    print(f"📊 原始数据量: {len(df)} 条")
    
    # 2. 数据清洗
    # 2.1 去重 (根据 instruction 和 input 两个字段判断是否重复)
    # 许多开源数据集会有重复的指令，去重能提高训练效率
    df_clean = df.drop_duplicates(subset=['instruction', 'input'])
    print(f"✂️ 去重后数据量: {len(df_clean)} 条 (删除了 {len(df) - len(df_clean)} 条重复数据)")
    
    # 2.2 去除空值 (确保关键字段不为空)
    df_clean = df_clean.dropna(subset=['instruction', 'output'])
    
    # 2.3 长度过滤 (清洗掉质量过差的数据)
    # 规则：回答长度必须大于 1 个字
    df_clean = df_clean[df_clean['output'].str.len() > 1]
    
    # 2.4 格式标准化 (可选)
    # 这里我们保持 instruction, input, output 的结构
    
    print(f"📉 过滤后最终数据量: {len(df_clean)} 条")
    
    # 3. 保存为 JSONL (JSON Lines) 格式
    # JSONL 格式：每一行是一个完整的 JSON 对象
    # 优点：支持流式读取，适合大规模数据集，不会一次性撑爆内存
    df_clean.to_json(output_file, orient='records', lines=True, force_ascii=False)
    
    print(f"✅ 清洗完成！已保存到: {output_file}")
    print("\n🔍 结果预览 (前2条):")
    
    # 预览文件内容
    with open(output_file, 'r', encoding='utf-8') as f:
        for i in range(2):
            print(f.readline().strip())

if __name__ == "__main__":
    clean_data()
