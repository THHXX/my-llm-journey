from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import json
import os

def train_custom_tokenizer():
    input_file = "processed_data.jsonl"
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件 {input_file}，请先运行 clean_data.py")
        return

    print("🚂 开始训练自定义 Tokenizer (BPE)...")
    
    # 1. 准备训练语料
    # Tokenizer 需要大量文本来学习词表。我们提取 instruction 和 output 字段。
    corpus_file = "tokenizer_corpus.txt"
    print("📝 正在生成语料文件...")
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(corpus_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            # 将指令和回答都作为训练语料
            f_out.write(data["instruction"] + "\n")
            f_out.write(data["output"] + "\n")
            
    print(f"📄 语料准备完成: {corpus_file}")

    # 2. 初始化 Tokenizer (使用 BPE 算法)
    # BPE (Byte-Pair Encoding) 是目前大模型最常用的分词算法
    tokenizer = Tokenizer(models.BPE())
    
    # 3. 预处理 (Pre-tokenization)
    # ByteLevel: 字节级处理，对代码和多语言支持更好（GPT-2/GPT-3/Llama 都在用）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 4. 配置训练器
    # vocab_size: 词表大小。
    # 对于演示项目，设为 10000 足够；商业大模型通常是 32000 - 100000+
    trainer = trainers.BpeTrainer(
        vocab_size=10000, 
        min_frequency=2, # 至少出现2次才会被收录
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )
    
    # 5. 开始训练
    print("⏳ 正在训练 Tokenizer，请稍候...")
    tokenizer.train([corpus_file], trainer)
    
    # 6. 后处理 (Decoding)
    # 解码时也需要 ByteLevel
    tokenizer.decoder = decoders.ByteLevel()
    
    # 7. 保存
    save_path = os.path.join(base_dir, "my_custom_tokenizer.json")
    tokenizer.save(save_path)
    print(f"✅ Tokenizer 训练完成并保存为 {save_path}")
    
    # 8. 测试
    print("\n🧪 效果测试:")
    test_texts = ["人工智能", "Hello World", "数据清洗很重要"]
    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"原文: {text}")
        print(f"Token IDs: {encoded.ids}")
        print(f"Tokens:    {encoded.tokens}")
        print("-" * 30)

if __name__ == "__main__":
    train_custom_tokenizer()
