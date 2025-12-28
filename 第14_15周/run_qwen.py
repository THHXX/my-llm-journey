import os
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 1. 极简主义：设置环境变量，防止不必要的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_inference():
    print("🚀 正在加载 Qwen-0.5B 模型 (从 ModelScope 国内源)...")
    
    # 2. 加载分词器 (Tokenizer) - 负责把人话变成数字
    # trust_remote_code=True 是必须的，因为 Qwen 的代码在远程仓库里
    # 修复：原 qwen/Qwen-0.5B 仓库已失效，改用最新的 Qwen2.5-0.5B-Instruct
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 3. 加载模型 (Model) - 负责计算和生成
    # device_map="cpu" 强制使用 CPU，确保消费级电脑也能跑
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    
    # 4. 准备输入
    prompt = "极简主义是什么？"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"\n👤 用户输入: {prompt}")
    print("🤖 正在思考中...\n")
    
    # 5. 生成回复
    # max_new_tokens=100: 限制生成长度，防止废话
    # do_sample=True: 让回答有点随机性，更像人
    pred = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    
    # 6. 解码输出 - 把数字变回人话
    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    print(f"💬 模型回答: {response}")

if __name__ == "__main__":
    run_inference()