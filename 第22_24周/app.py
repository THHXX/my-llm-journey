import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rag_engine import LawRAG
import os
import torch


# 禁用代理以解决 Gradio 启动时的 502/Connection Refused 错误
# os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# --- 1. 初始化路径 ---
# 假设 lora_output 在 第18_19周 目录下
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../第18_19周/lora_output"))
BASE_MODEL = r"C:\Users\JYJYJ\.cache\huggingface\hub\models--qwen--Qwen1.5-0.5B\snapshots\8f445e3628f3500ee69f24e1303c9f10f5342a39"

# --- 2. 加载模型 ---
print(f"🚀 正在加载基础模型: {BASE_MODEL}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # 强制使用 CPU 运行，避免显存问题，虽然慢一点但稳
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu", trust_remote_code=True)
    
    # 加载 LoRA 权重
    if os.path.exists(LORA_PATH):
        print(f"✅ 挂载 LoRA 权重: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    else:
        print(f"⚠️ 未找到 LoRA 权重 ({LORA_PATH})，将使用基础模型运行！")
        
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 为了演示 UI，这里不退出，但后续会报错

# --- 3. 初始化 RAG ---
kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
if os.path.exists(kb_path):
    rag = LawRAG(kb_path)
else:
    print("⚠️ 未找到知识库文件，RAG 功能将失效。")
    rag = None

# --- 4. 定义对话函数 ---
def chat_response(message, history):
    context = ""
    
    # Step A: 检索知识
    if rag:
        try:
            retrieved_docs = rag.search(message)
            context = "\n".join(retrieved_docs)
            print(f"🔍 RAG 检索结果: {retrieved_docs}")
        except Exception as e:
            print(f"❌ RAG 检索出错: {e}")
    
    # Step B: 构建 Prompt
    # 极简 Prompt 模板
    prompt = f"""你是一个严谨的法律助手。请完全依据下面的参考资料回答问题。如果参考资料里没有答案，请直接说“我不知道”，不要编造。

参考资料：
{context}

用户问题：{message}

回答："""

    # Step C: 模型推理
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 生成
        pred = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.7,
            do_sample=True
        )
        
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        
        # 简单清洗：尝试去掉 prompt 部分，只保留回答
        # 这里用一个简单的 split 策略，实际可能需要更复杂的处理
        if "回答：" in response:
            response = response.split("回答：")[-1]
        elif "Output:" in response: # 兼容训练时的格式
            response = response.split("Output:")[-1]
            
        return response.strip()
        
    except Exception as e:
        return f"模型推理出错: {str(e)}"

# --- 5. 启动界面 ---
demo = gr.ChatInterface(
    fn=chat_response,
    title="⚖️ AI 法律顾问 (RAG + LoRA)",
    description="基于 Qwen-0.5B 微调，挂载《民法典》知识库。请输入法律问题，例如“离婚需要什么条件？”",
    examples=["离婚需要什么条件？", "对方出轨了怎么办？", "抚养费怎么算？"]
)

if __name__ == "__main__":
    demo.launch()
