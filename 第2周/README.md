# Qwen API 调用示例（第二周）

## 1. 获取阿里云百炼 API Key
1. 打开控制台：https://bailian.console.aliyun.com/
2. 完成实名认证（若提示）。
3. 创建应用，选择文本大模型，生成 API Key。
4. 复制密钥，后续在本地以环境变量 `DASHSCOPE_API_KEY` 使用。

## 2. 本地环境准备（PowerShell，每条命令都含注释）
```pwsh
python -m venv .venv  # 创建虚拟环境（可选但推荐）
.\.venv\Scripts\Activate.ps1  # 激活虚拟环境
python -m pip install --upgrade pip  # 升级 pip
python -m pip install -r requirements.txt  # 安装依赖（包含 dashscope）
```

> 若尚未安装 Python，请先在 https://www.python.org/downloads/ 下载并勾选 “Add to PATH”。

## 3. 配置 API Key
### 临时（当前会话有效）
```pwsh
$env:DASHSCOPE_API_KEY="sk-xxx"  # 替换为你的百炼密钥
```

### 永久（写入用户环境变量，重开终端生效）
```pwsh
setx DASHSCOPE_API_KEY "sk-xxx"  # 替换为你的百炼密钥
```

## 4. 运行示例脚本
```pwsh
python qwen_call.py  # 使用默认提示词调用模型
python qwen_call.py 请用50字介绍一下量子计算  # 传入自定义提示
```

运行成功将输出：
- 成功：打印 “模型回复：” 及生成文本。
- 失败：在标准错误输出报错信息（常见为密钥未配置或配额不足）。

## 5. GitHub 提交提示
```pwsh
git status  # 查看变更
git add 第二周  # 暂存新增文件
git commit -m "Add Qwen API call example"  # 提交
git push  # 推送到远端
```


