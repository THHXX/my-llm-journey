import os
import gradio as gr
from dashscope import Generation
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from http import HTTPStatus

# ========== 配置部分 ==========
# 1. 设置 HF 镜像，防止模型下载超时
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HUGGINGFACE_SPACES" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# MODEL_PATH = r"C:\Users\JYJYJ\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
# 替换为中文效果更好的 Embedding 模型
# 第一次运行时会自动下载（约 100MB），请确保网络通畅
MODEL_PATH = "BAAI/bge-small-zh-v1.5"
# MODEL_PATH = r"C:\Users\JYJYJ\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620"
# 全局变量：用于存储构建好的向量库
global_vectorstore = None

# ========== 核心逻辑 ==========

def load_and_split_files(file_objs):
    """
    加载并切分上传的文件。
    支持 PDF 和 TXT。
    """
    all_chunks = []
    
    if not file_objs:
        return []

    for file_obj in file_objs:
        # Gradio 4.x 传入的是文件对象列表，file_obj.name 是临时文件路径
        file_path = file_obj.name
        filename = os.path.basename(file_path)
        print(f"[System] 正在处理文件: {filename}")
        
        # 根据后缀选择加载器
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        else:
            print(f"[System] 跳过不支持的文件格式: {filename}")
            continue

        docs = loader.load()
        
        # 切分文本
        # 针对中文优化切分参数
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 减小切分粒度，提高检索精准度
            chunk_overlap=50 # 减少重叠
        )
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        
    return all_chunks

def build_vectorstore(file_objs):
    """
    根据上传的文件构建向量库。
    """
    global global_vectorstore
    
    chunks = load_and_split_files(file_objs)
    if not chunks:
        return "⚠️ 未提取到有效文本，请检查文件格式。"

    print(f"[System] 共切分出 {len(chunks)} 个文本块，开始构建向量库...")
    
    # 初始化 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    
    # 构建 FAISS 索引
    global_vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return f"✅ 知识库构建完成！共处理 {len(chunks)} 个片段。现在可以开始提问了。"

def qwen_chat(message, history):
    """
    处理用户提问（支持多轮对话）。
    message: 当前用户输入
    history: 历史对话列表 [[user_msg, bot_msg], ...]
    返回: 生成的回答文本
    """
    global global_vectorstore
    
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return "❌ 请先设置 DASHSCOPE_API_KEY 环境变量。"
    
    print(f"[DEBUG] message type: {type(message)}")
    print(f"[DEBUG] message content: {message}")

    # 防御性处理：如果 message 是列表，取第一个元素
    if isinstance(message, list):
        if len(message) > 0:
            message = message[0]
            print(f"[DEBUG] Converted list message to: {message}")
        else:
            message = ""
    
    # 确保 message 是字符串
    if not isinstance(message, str):
        message = str(message)

    if global_vectorstore is None:
        return "⚠️ 请先上传文件并等待知识库构建完成。"

    # 1. 检索相关文档
    try:
        # 增加检索数量 k=5，提供更多上下文
        retrieved_docs = global_vectorstore.similarity_search(message, k=5)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        return f"检索出错: {str(e)}"

    # 2. 构建 Prompt
    history_str = ""
    # 历史记录格式为 [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    # 取最近 4 条消息（2轮对话）
    recent_history = history[-4:] if len(history) >= 4 else history
    for msg in recent_history:
        role = "用户" if msg['role'] == "user" else "助手"
        history_str += f"{role}: {msg['content']}\n"

    prompt = f"""
你是一个专业的知识库助手。请根据以下参考资料回答用户问题。

【参考资料】
{context}

【历史对话】
{history_str}

【用户当前问题】
{message}

请注意：
1. 请务必仅依据【参考资料】中的内容回答，不要使用你自己的外部知识。
2. 如果【参考资料】中没有包含问题的答案，请直接说“抱歉，根据提供的资料，我无法回答这个问题”，不要编造答案。
3. 回答要条理清晰，使用中文，并尽量引用资料中的原文。
4. 请一步步思考，确保逻辑严密。
"""

    # 3. 调用 Qwen API
    try:
        response = Generation.call(
            model="qwen-max",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            prompt=prompt
        )
        
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        else:
            return f"API 调用失败: {response.message}"
            
    except Exception as e:
        return f"发生错误: {str(e)}"

def user_input(user_msg, history):
    """用户输入处理：将消息添加到历史记录，并清空输入框"""
    if history is None:
        history = []
    # 强制使用 OpenAI 格式，满足 Gradio 报错要求
    return "", history + [{"role": "user", "content": user_msg}]

def bot_response(history):
    """机器人响应处理：获取最后一条用户消息，调用模型，更新历史记录"""
    # 获取最后一条用户消息的内容
    user_msg = history[-1]['content']
    # 调用模型生成回答，传入除当前问题外的历史记录
    bot_msg = qwen_chat(user_msg, history[:-1])
    # 添加机器人回答
    history.append({"role": "assistant", "content": bot_msg})
    return history

# ========== Gradio 界面 ==========

with gr.Blocks(title="个人知识库助手") as demo:
    gr.Markdown("# 📚 个人 RAG 知识库助手")
    gr.Markdown("支持上传 PDF/TXT 文件，构建专属知识库并进行问答。")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 文件上传区
            file_input = gr.File(
                label="1. 上传文档 (支持多选)", 
                file_count="multiple",
                file_types=[".pdf", ".txt"]
            )
            upload_btn = gr.Button("🚀 构建知识库", variant="primary")
            status_output = gr.Textbox(label="系统状态", interactive=False)
            
        with gr.Column(scale=2):
            # 聊天区
            # 不传 type 参数，但数据格式改为字典列表，以满足报错提示的要求
            chatbot = gr.Chatbot(height=500, label="对话记录")
            msg_input = gr.Textbox(label="2. 输入问题", placeholder="关于文档内容的提问...")
            clear_btn = gr.Button("🗑️ 清空对话")

    # 事件绑定
    upload_btn.click(
        fn=build_vectorstore,
        inputs=[file_input],
        outputs=[status_output]
    )
    
    # 提交问题后的处理流程：
    # 1. user_input: 更新 Chatbot 显示用户问题，清空输入框
    # 2. bot_response: 调用模型生成回答，更新 Chatbot 显示助手回答
    msg_input.submit(
        fn=user_input,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot]
    ).then(
        fn=bot_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    # 清空按钮
    clear_btn.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch()