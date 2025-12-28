"""
第5-6周：RAG 系统实验（步骤1+2+3+4）

本文件目前实现：
1. 使用 PyPDFLoader 加载本地 PDF 文档
2. 用 RecursiveCharacterTextSplitter 把文档切成很多小块
3. 使用 HuggingFaceEmbeddings + FAISS 把所有小块构建成向量库
4. 用一句“这篇文章主要讲了什么？”做一次相似度检索测试
5. 调用 Qwen 模型，基于检索到的内容生成摘要答案（完整 RAG 流程）
"""

import os  # 读取环境变量中的 API Key
import sys  # 打印错误信息并在必要时退出程序
from http import HTTPStatus  # 判断 HTTP 返回状态码是否为 200
from typing import Optional  # 为函数返回值添加可选类型注解

from langchain_community.document_loaders import PyPDFLoader  # 从 PDF 加载文档
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 切分长文本
from langchain_community.embeddings import HuggingFaceEmbeddings  # 使用 sentence-transformers 生成向量
from langchain_community.vectorstores import FAISS  # 使用 FAISS 构建向量库
from dashscope import Generation  # 调用通义千问 Qwen 的 Python SDK


def load_docs_from_pdf() -> list:
    """使用 PyPDFLoader 从本地 PDF 文件加载文档。"""
    pdf_path = "./第5_6周/1.pdf"  # TODO: 把这里改成你自己 PDF 的相对路径
    loader = PyPDFLoader(pdf_path)  # 创建 PDF 加载器
    docs = loader.load()  # 读取 PDF，返回一个 Document 列表
    return docs


def split_docs(docs: list) -> list:
    """使用递归字符切分器，把长文档切成很多小块。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每块大约 500 个字符
        chunk_overlap=100,   # 相邻块之间重叠 100 个字符，保持上下文连续
    )
    chunks = text_splitter.split_documents(docs)  # 执行切分，返回新的 Document 列表
    return chunks


def build_vectorstore(chunks: list):
    """
    使用 HuggingFaceEmbeddings + FAISS 构建向量库。
    chunks: 已经切分好的 Document 列表，每个都有 .page_content 文本。
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 复用第4周已经下载好的句向量模型
    embeddings = HuggingFaceEmbeddings(model_name=model_name)  # 创建向量模型（本地计算向量）

    # from_documents 会自动：
    # 1. 调用 embeddings 把每个 chunk 的文本编码成向量
    # 2. 把向量 + 对应的 Document 一起存进 FAISS 索引
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore  # 返回向量库对象，后面用来做相似度检索


def test_similarity_search(vectorstore):
    """
    用一条示例问题，在向量库里检索最相似的文档块。
    目的：检查切分 + 向量化是否合理。
    """
    query = "这篇文章主要讲了什么？"  # 示例问题，你可以改成更贴合你 PDF 的问题
    k = 3  # 检索出最相近的 3 个块

    # similarity_search 内部流程：
    # 1. 用同一个 embeddings 模型把 query 编码成向量
    # 2. 在 FAISS 向量库里找与 query 向量最接近的 k 个文本块
    results = vectorstore.similarity_search(query, k=k)

    print(f"检索问题: {query}")
    print(f"返回的文档块数量: {len(results)}")
    for i, doc in enumerate(results):
        print("=" * 40)
        print(f"第 {i + 1} 个相似块：")  # 打印当前是第几个块
        print(doc.page_content[:300])  # 预览前 300 个字符，避免输出太长


def run_qwen_generation(prompt: str, api_key: str) -> Optional[str]:
    """
    调用通义千问 Qwen（dashscope）生成答案的封装函数。
    prompt: 已经拼接好的完整提示词（包含上下文和问题）
    api_key: 从环境变量中读取的 DASHSCOPE_API_KEY
    """
    try:
        response = Generation.call(  # 调用 Qwen 的文本生成接口
            model="qwen-max",  # 使用较强能力的 qwen-max 模型
            api_key=api_key,  # 传入 API Key 进行鉴权
            prompt=prompt,  # 把我们构造好的 Prompt 传给模型
        )
    except Exception as exc:  # 捕获网络错误或 SDK 异常
        print(f"调用 Qwen 异常: {exc}", file=sys.stderr)
        return None  # 返回 None 表示调用失败

    if response.status_code == HTTPStatus.OK:  # 判断 HTTP 状态是否为 200
        return response.output.text  # 正常则返回模型生成的文本

    # 如果不是 200，则打印错误信息，方便排查
    print(
        f"调用 Qwen 失败，状态码: {response.status_code}, 错误信息: {response.message}",
        file=sys.stderr,
    )
    return None  # 返回 None 表示失败


def answer_question_with_rag(vectorstore, question: str) -> None:
    """
    使用 RAG 流程回答问题：
    1. 先用向量库检索与问题最相关的文档块（Retrieval）
    2. 再把这些文档块作为上下文，交给 Qwen 生成答案（Generation）
    """
    # 第一步：从向量库中检索最相关的文档块
    k = 4  # 检索 4 个文档块，信息更充分一些
    retrieved_docs = vectorstore.similarity_search(question, k=k)  # 执行相似度检索

    # 把多个文档块的内容拼接成一个长上下文字符串，并用分隔线分开，方便阅读
    context = "\n\n------ 文档块分隔线 ------\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # 构造给 Qwen 的 Prompt，明确说明：只能根据提供的内容回答
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

    # 从环境变量中读取 Qwen 的 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")  # 读取 DASHSCOPE_API_KEY 环境变量
    if not api_key:  # 如果没有配置，则给出提示并返回
        print(
            "未找到环境变量 DASHSCOPE_API_KEY，请先配置你的 Qwen API Key。",
            file=sys.stderr,
        )
        return

    # 调用 Qwen 生成最终答案
    answer = run_qwen_generation(prompt, api_key)  # 发送请求并获取返回结果
    if answer is None:  # 如果调用失败，直接结束
        print("Qwen 调用失败，无法生成答案。", file=sys.stderr)
        return

    # 打印最终回答，作为 RAG 流程的输出
    print("\n=== 基于 RAG 的 Qwen 回答 ===")
    print(answer)


def main():
    """加载 PDF -> 切分 -> 构建向量库 -> 做一次检索测试 -> 调用 Qwen 回答。"""
    docs = load_docs_from_pdf()  # 第一步：加载原始 PDF 文档
    print(f"原始文档数量: {len(docs)}")  # 打印原始 Document 数量（一般是按页计）

    chunks = split_docs(docs)  # 第二步：把长文档切成小块
    print(f"切分后的文档块数量: {len(chunks)}")  # 打印切分后块数

    vectorstore = build_vectorstore(chunks)  # 第三步：用 embedding + FAISS 构建向量库

    test_similarity_search(vectorstore)  # 第四步：用一个问题测试检索效果（检查检索是否合理）

    # 第五步：真正用 RAG + Qwen 回答同一个问题，生成自然语言摘要
    question = "这篇文章主要讲了什么？"  # 可以根据自己的 PDF 内容修改问题
    answer_question_with_rag(vectorstore, question)  # 调用 RAG 问答函数


if __name__ == "__main__":
    main()  # 运行主流程