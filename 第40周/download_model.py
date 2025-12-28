import os
from modelscope import snapshot_download

# 定义模型保存路径
model_dir = os.path.join(os.getcwd(), "model", "Qwen1.5-0.5B")

print(f"🚀 开始下载 Qwen1.5-0.5B 模型到: {model_dir}")

# 从 ModelScope 下载
model_path = snapshot_download(
    'qwen/Qwen1.5-0.5B', 
    cache_dir=os.path.join(os.getcwd(), "model_cache"),
    local_dir=model_dir
)

print(f"✅ 模型下载完成！路径: {model_path}")
print("下一步：请参考 操作.md 进行格式转换和量化。")
