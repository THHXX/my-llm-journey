import os
import time
import sys

# -------------------------------------------------------------------------
# 用户指定模型路径 (Windows 格式)
# -------------------------------------------------------------------------
ORIGIN_MODEL_PATH = r"C:\Users\JYJYJ\.cache\huggingface\hub\models--qwen--Qwen1.5-0.5B\snapshots\8f445e3628f3500ee69f24e1303c9f10f5342a39"

def get_wsl_path(win_path):
    """如果是在 WSL2 环境下，自动将 Windows 路径转换为 Linux 路径"""
    if sys.platform.startswith('linux'):
        # 简单判断是否为 C 盘路径并转换
        if "C:" in win_path:
            return win_path.replace("C:", "/mnt/c").replace("\\", "/")
    return win_path

# 获取最终路径
model_path = get_wsl_path(ORIGIN_MODEL_PATH)

# -------------------------------------------------------------------------
# vLLM 检测与加载
# -------------------------------------------------------------------------
try:
    from vllm import LLM, SamplingParams
    VLLM_INSTALLED = True
except ImportError:
    VLLM_INSTALLED = False
    print("❌ 未检测到 vLLM 库。")
    print("原因：当前是 Windows 原生环境，requirements.txt 已自动跳过安装。")
    print("--------------------------------------------------")
    print("💡 解决方案 (二选一)：")
    print("1. [推荐] 放弃 vLLM，改用 llama.cpp (GGUF) 运行量化模型。")
    print("2. [死磕] 请安装 WSL2 (Ubuntu)，在 Linux 环境中运行此代码。")
    print("--------------------------------------------------")

if VLLM_INSTALLED:
    print(f"🚀 正在使用 vLLM 加载模型: {model_path}")
    print("注意：显存需大于 4GB，否则请调低 gpu_memory_utilization")

    try:
        # 初始化 vLLM (gpu_memory_utilization 控制显存占用，0.6 表示占用 60%)
        llm = LLM(model=model_path, quantization=None, gpu_memory_utilization=0.6)
        
        # 定义采样参数
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=128)

        # 测试 Prompts
        prompts = [
            "Hello, my name is",
            "为什么天空是蓝色的？",
        ]

        print("⚡ 开始推理...")
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        end_time = time.time()

        # 输出结果
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt}\nGenerated: {generated_text}")
            
        print(f"\n✅ 推理完成！耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"⚠️ vLLM 运行出错: {e}")