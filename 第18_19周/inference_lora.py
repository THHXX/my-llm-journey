import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

def inference():
    # 路径配置
    base_model_path = "qwen/Qwen1.5-0.5B"
    lora_path = os.path.join(os.path.dirname(__file__), "lora_output")
    
    print(f"🚀 加载基础模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", trust_remote_code=True)
    
    print(f"🔗 加载 LoRA 权重: {lora_path}")
    # 加载微调后的权重
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as e:
        print(f"❌ 加载 LoRA 失败: {e}")
        print("💡 请先运行 train_lora.py 完成训练")
        return

    model.eval()
    
    # 测试问题
    test_questions = [
        "什么是合同法？",
        "甲方违约了怎么办？",
        "医生有告知义务吗？", # 训练集里的问题
        "解释一下不可抗力。" # 稍微变体
    ]
    
    print("\n💬 开始对话测试:")
    print("="*50)
    
    for q in test_questions:
        prompt = f"Instruction: {q}\nInput: \nOutput: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False # 设为 False 方便复现
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 Output 后面的内容
        try:
            answer = response.split("Output: ")[1].strip()
        except:
            answer = response
            
        print(f"❓ 问: {q}")
        print(f"🤖 答: {answer}")
        print("-" * 30)

if __name__ == "__main__":
    inference()
