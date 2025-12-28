import os
import subprocess
import time
import requests
import psutil
import sys

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 (第43_48周)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 路径配置
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools", "llama-cpp")
SERVER_EXE = os.path.join(TOOLS_DIR, "llama-server.exe")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "qwen1.5-0.5b-chat-q4_k_m.gguf")

HOST = "127.0.0.1"
PORT = 8080
API_URL = f"http://{HOST}:{PORT}/v1/chat/completions"

class LocalLLM:
    def __init__(self, port=8080):
        self.port = port
        self.process = None
        self.ensure_server_running()

    def is_port_in_use(self):
        """检查端口是否被占用"""
        for conn in psutil.net_connections():
            if conn.laddr.port == self.port:
                return True
        return False

    def ensure_server_running(self):
        """确保 llama-server 正在运行"""
        if self.is_port_in_use():
            print(f"   ✅ LLM 服务似乎已在端口 {self.port} 运行")
            # 可以在这里做一个简单的健康检查
            return

        print(f"   🚀 正在启动本地 LLM 服务 (Port {self.port})...")
        print(f"   📂 模型路径: {MODEL_PATH}")
        
        if not os.path.exists(SERVER_EXE):
            raise FileNotFoundError(f"找不到 llama-server.exe: {SERVER_EXE}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")

        # 启动命令
        # 尝试使用相对路径或切换工作目录以避免中文路径问题
        model_filename = os.path.basename(MODEL_PATH)
        
        cmd = [
            SERVER_EXE,
            "-m", model_filename,
            "-c", "2048",
            "--host", HOST,
            "--port", str(self.port),
            "-ngl", "0" 
        ]

        # 日志文件
        log_file = open(os.path.join(PROJECT_ROOT, "llama_server.log"), "w")
        
        self.process = subprocess.Popen(
            cmd,
            cwd=MODELS_DIR, # 切换工作目录到模型所在目录
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
        # 等待服务启动
        print("   ⏳ 等待服务就绪...", end="", flush=True)
        retries = 30
        for _ in range(retries):
            if self.is_port_in_use():
                print(" 完成!")
                return
            time.sleep(1)
            print(".", end="", flush=True)
        
        raise RuntimeError("LLM 服务启动超时")

    def chat(self, messages, temperature=0.7):
        """
        发送聊天请求
        messages: [{"role": "user", "content": "..."}]
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
            "frequency_penalty": 1.1, # 增加重复惩罚
            "presence_penalty": 1.1,
            "top_p": 0.95
        }
        
        # 重试机制
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, json=payload, timeout=60)
                
                # 如果是 503 (模型正在加载中)，等待并重试
                if response.status_code == 503:
                    print(f"   ⏳ 服务繁忙或正在初始化 (503)，重试 {attempt+1}/{max_retries}...")
                    time.sleep(2)
                    continue
                    
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"\n❌ 请求 LLM 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None
        return None

    def stop(self):
        """停止服务"""
        if self.process:
            self.process.terminate()
            self.process = None

if __name__ == "__main__":
    # 测试代码
    try:
        llm = LocalLLM()
        
        test_msg = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
        print(f"\n🗣️  发送测试消息: {test_msg[0]['content']}")
        
        result = llm.chat(test_msg)
        
        if result:
            content = result['choices'][0]['message']['content']
            print(f"\n🤖 回复:\n{content}")
        
        # 保持服务运行，或者选择 llm.stop()
        # llm.stop() 
        print("\n✅ 测试完成。服务仍在后台运行，可供 RAG 使用。")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
