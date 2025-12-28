"""
第7-8周：基于 Gradio 的 RAG Web 应用（上传 PDF + 提问）

功能：
1. 用户在网页上传一份 PDF，并输入问题（例如：这篇文章讲了什么？）
2. 后端对 PDF 做 RAG 流程：加载 -> 切分 -> 向量化 -> 检索 -> 调用 Qwen 生成答案
3. 将答案显示在网页上

注意：
- 这是一个「教学版」示例，每次提问都会重新构建向量库，因此速度较慢，但逻辑清晰。
- 后续可以做缓存优化（例如对同一个文件只构建一次向量库）。
"""

import os  # 读取环境变量中的 API Key
import sys  # 打印错误信息
from http import HTTPStatus  # 判断 HTTP 状态码
from typing import Optional  # 可选类型注解

import gradio as gr  # Gradio：快速搭建 Web 界面

# ========== 模型路径配置 ==========
# 请将下方路径修改为您本地下载的模型文件夹绝对路径
# 例如：MODEL_PATH = r"C:\Users\JYJYJ\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2"
# 如果没有本地模型，保持原样将尝试从 Hugging Face 下载
MODEL_PATH = r"C:\Users\JYJYJ\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

from dashscope import Generation  # 调用通义千问 Qwen 的 Python SDK
from langchain_community.document_loaders import PyPDFLoader  # 从 PDF 加载文档
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本切分
from langchain_community.embeddings import HuggingFaceEmbeddings  # 生成文本向量
from langchain_community.vectorstores import FAISS  # 向量库 FAISS


# ========== 一、Qwen 调用封装 ==========

def run_qwen_generation(prompt: str, api_key: str) -> Optional[str]:
    """
    调用通义千问 Qwen（dashscope）生成答案的封装函数。
    prompt: 已经拼接好的完整提示词（包含上下文和问题）
    api_key: 从环境变量中读取的 DASHSCOPE_API_KEY
    """
    try:
        response = Generation.call(
            model="qwen-max",  # 使用 qwen-max 模型
            api_key=api_key,  # API Key
            prompt=prompt,  # 提示词
        )
    except Exception as exc:
        print(f"调用 Qwen 异常: {exc}", file=sys.stderr)
        return None

    if response.status_code == HTTPStatus.OK:
        return response.output.text  # 返回生成的文本

    print(
        f"调用 Qwen 失败，状态码: {response.status_code}, 错误信息: {response.message}",
        file=sys.stderr,
    )
    return None


# ========== 二、RAG 核心逻辑：PDF -> 向量库 -> 检索 -> 总结 ==========

def build_vectorstore_from_pdf(pdf_path: str):
    """
    从指定 PDF 文件路径构建向量库（FAISS）。
    步骤：加载 PDF -> 文本切分 -> embedding -> FAISS 向量库。
    """
    # 1. 加载 PDF 为 Document 列表
    loader = PyPDFLoader(pdf_path)  # 创建 PDF 加载器
    docs = loader.load()  # 读取 PDF
    print(f"[RAG] 原始文档页数: {len(docs)}")

    # 2. 文本切分：把较长的文档拆成多个小块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # 每块大约 500 个字符
        chunk_overlap=100  # 块之间重叠 100 字符，保留上下文
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] 切分后的文档块数量: {len(chunks)}")

    # 3. 构建向量库：使用 sentence-transformers 模型生成向量
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 第4周已使用的句向量模型
    print(f"[RAG] 正在加载模型: {MODEL_PATH} ...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)  # 创建 embedding 模型

    vectorstore = FAISS.from_documents(chunks, embeddings)  # 构建 FAISS 向量库
    return vectorstore


def rag_answer_question(pdf_path: str, question: str) -> str:
    """
    对给定的 PDF 文件和问题执行完整 RAG 流程，返回最终答案字符串。
    """
    # 1. 从 PDF 构建向量库（内部包含加载、切分和 embedding）
    vectorstore = build_vectorstore_from_pdf(pdf_path)

    # 2. 使用向量库检索与问题最相关的文档块
    k = 4  # 检索前 4 个相关块
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    if not retrieved_docs:
        return "未能从文档中检索到相关内容，请检查 PDF 是否有有效文本。"

    # 3. 将检索到的多个文档块拼接成上下文
    context = "\n\n------ 文档块分隔线 ------\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # 4. 构造 Prompt：告诉 Qwen 上下文 + 问题 + 回答要求
    prompt = f"""
你是一个严谨的阅读理解助手。请严格按照下面的要求回答问题。

【已检索到的文章片段】
{context}

【用户问题】
{question}

【回答要求】
1. 你的回答必须只基于【已检索到的文章片段】中的信息，不要自己编造内容。
2. 请用中文给出这篇文章的简要摘要，控制在 100~150 字左右。
3. 如果片段中的信息不足以回答问题，请明确说明“根据当前片段信息不足，无法给出完整答案”。
""".strip()

    # 5. 读取 Qwen API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "未找到环境变量 DASHSCOPE_API_KEY，请先在终端中设置你的 Qwen API Key。"

    # 6. 调用 Qwen 生成答案
    answer = run_qwen_generation(prompt, api_key)
    if answer is None:
        return "调用 Qwen 失败，请检查网络或 API Key 是否正确。"

    return answer


# ========== 三、Gradio 回调函数：供网页调用 ==========

def qa(question: str, file) -> str:
    """
    Gradio 接口函数：
    - question: 用户在网页输入的问题
    - file: 用户上传的 PDF 文件（Gradio 传入的对象，其中包含文件路径）
    """
    if not file:
        return "请先上传一份 PDF 文件。"

    if not question or not question.strip():
        return "请输入一个问题，例如：这篇文章讲了什么？"

    # 从 Gradio 传入的 file 对象中获取真实文件路径
    # 新版 gradio 会传一个类似 tempfile 对象，通常有 .name 属性
    if hasattr(file, "name"):
        pdf_path = file.name  # 从对象中取出实际的临时文件路径
    else:
        # 有些版本可能直接传字符串路径，这里做一个兼容处理
        pdf_path = str(file)

    print(f"[Web] 收到问题: {question}")
    print(f"[Web] 上传文件路径: {pdf_path}")

    # 调用上面的 RAG 流程获取答案
    answer = rag_answer_question(pdf_path, question)
    return answer  # 返回给 Gradio，在网页上展示


# ========== 四、构建并启动 Gradio 界面 ==========

def main():
    """构建 Gradio 界面并启动本地 Web 服务。"""
    # 输入组件：
    # - 一个多行文本框用于输入问题
    # - 一个文件上传控件用于上传 PDF
    question_input = gr.Textbox(
        label="请输入你的问题",
        placeholder="例如：这篇文章讲了什么？",
        lines=2,
    )
    file_input = gr.File(
        label="上传 PDF 文件",
        file_types=[".pdf"],  # 只接受 PDF
    )

    # 输出组件：一个多行文本框显示答案
    answer_output = gr.Textbox(
        label="回答",
        lines=8,
    )

    # 使用 Interface 快速构建应用
    demo = gr.Interface(
        fn=qa,  # 绑定后端处理函数
        inputs=[question_input, file_input],  # 输入组件列表
        outputs=answer_output,  # 输出组件
        title="第7-8周：RAG 问答 Demo",  # 页面标题
        description="上传一份 PDF，输入一个问题（如：这篇文章讲了什么？），系统会基于文档内容进行检索并由 Qwen 生成摘要回答。",
    )

    # 启动本地 Web 服务
    demo.launch()  # 默认在 http://localhost:7860 启动


if __name__ == "__main__":
    main()  # 运行主函数，启动 Gradio 应用