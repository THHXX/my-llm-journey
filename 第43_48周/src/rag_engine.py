import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_parser import extract_text_from_pdf
from llm_interface import LocalLLM

# 配置路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")
CHROMA_DB_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "chroma_db")

def build_vector_db(pdf_filename="tesla_2023_10k.pdf", collection_name="financial_reports"):
    """
    构建向量数据库：读取PDF -> 切分 -> 嵌入 -> 存入 ChromaDB
    """
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到文件 {pdf_path}")
        return None

    # 1. 提取文本
    print(f"1. 正在读取 {pdf_filename} ...")
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        print("提取文本失败。")
        return None
    print(f"   提取成功，共 {len(full_text)} 字符。")

    # 2. 文本切分 (Chunking)
    # 使用 RecursiveCharacterTextSplitter 智能切分
    print("2. 正在进行文本切分...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 每个块约 1000 字符
        chunk_overlap=200,    # 重叠 200 字符，防止上下文丢失
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)
    print(f"   切分完成，共生成 {len(chunks)} 个文本块。")

    # 初始化 ChromaDB 客户端
    print(f"3. 初始化 ChromaDB (持久化路径: {CHROMA_DB_DIR})...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # -----------------------------------------------------------
    # 使用 ModelScope (阿里云) 下载模型，解决国内网络问题
    # -----------------------------------------------------------
    # 【重要】设置缓存目录到 D 盘，避免占用 C 盘空间
    os.environ['MODELSCOPE_CACHE'] = 'D:\\ModelScope_Cache'
    
    model_name_or_path = "all-MiniLM-L6-v2" # 默认值
    try:
        from modelscope import snapshot_download
        print("   🚀 正在使用 ModelScope (阿里云) 下载模型...")
        print(f"   📂 缓存目录: {os.environ['MODELSCOPE_CACHE']}")
        # 这里的 model_id 是 ModelScope 上的镜像 ID
        model_dir = snapshot_download('AI-ModelScope/all-MiniLM-L6-v2', revision='master')
        model_name_or_path = model_dir
        print(f"   ✅ 模型已下载至: {model_dir}")
    except Exception as e:
        print(f"   ⚠️ ModelScope 下载失败，尝试直接加载 ({e})")

    # 使用 sentence-transformers 加载本地模型
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name_or_path
    )

    # 获取或创建集合 (Collection)
    # 如果已存在，先删除重建 (为了测试方便，实际生产可以增量更新)
    try:
        # 先检查集合是否存在，如果存在则删除
        try:
            client.get_collection(name=collection_name)
            client.delete_collection(name=collection_name)
            print(f"   已删除旧集合 {collection_name}")
        except:
            pass # 集合不存在，无需删除
    except Exception as e:
        print(f"   清理集合时出错: {e}")

    collection = client.create_collection(
        name=collection_name,
        embedding_function=emb_fn
    )

    # 4. 批量存入数据
    print("4. 正在生成向量并存入数据库 (这可能需要几分钟)...")
    
    # 构造 ID 和元数据
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_filename, "chunk_index": i} for i in range(len(chunks))]
    
    # ChromaDB 建议分批插入，防止一次性过大
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        print(f"   正在处理批次 {i//batch_size + 1}/{total_batches} (Chunk {i} - {batch_end})...")
        collection.add(
            documents=chunks[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        
    print("✅ 向量数据库构建完成！")
    return collection

def query_vector_db(query_text, collection_name="financial_reports", n_results=3):
    """
    查询向量数据库
    """
    print(f"\n🔎 正在查询: '{query_text}' ...")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # -----------------------------------------------------------
    # 同样使用 ModelScope 路径加载模型
    # -----------------------------------------------------------
    os.environ['MODELSCOPE_CACHE'] = 'D:\\ModelScope_Cache'
    model_name_or_path = "all-MiniLM-L6-v2"
    try:
        from modelscope import snapshot_download
        # 此时应该已经缓存了，不会重复下载
        model_dir = snapshot_download('AI-ModelScope/all-MiniLM-L6-v2', revision='master')
        model_name_or_path = model_dir
    except:
        pass

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name_or_path
    )
    
    try:
        collection = client.get_collection(name=collection_name, embedding_function=emb_fn)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        print(f"   找到 {len(results['documents'][0])} 个相关片段:\n")
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            print(f"   [片段 {i+1}] (来源: {meta['source']}, Index: {meta['chunk_index']})")
            print(f"   {'-'*50}")
            print(f"   {doc[:200]}...") # 只显示前200字符
            print(f"   {'-'*50}\n")
            
    except Exception as e:
        print(f"查询出错: {e}")
        return []
    
    return results['documents'][0]

def rag_chat(query_text, collection_name="financial_reports"):
    """
    RAG 对话：检索 -> 生成
    """
    # 1. 检索相关文档
    retrieved_docs = query_vector_db(query_text, collection_name, n_results=3)
    
    if not retrieved_docs:
        print("未找到相关文档，无法回答。")
        return

    # 2. 构建 Prompt (针对小模型优化：简洁指令)
    context = "\n\n".join(retrieved_docs)
    
    system_prompt = """You are a Financial Analyst. Analyze the Context to answer the Question.
    
Rules:
1. Only use the provided Context.
2. If the answer is not in Context, say "Data not available".
3. Keep answers concise and professional.
4. Use bullet points for lists.
"""
    
    user_prompt = f"""
Context:
{context}

Question: 
{query_text}

Answer:
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 3. 调用 LLM
    print(f"\n🤖 正在请求 LLM 生成回答...")
    try:
        llm = LocalLLM() # 会自动确保服务运行
        response = llm.chat(messages, temperature=0.1) # 低温度以保证基于事实
        
        if response:
            answer = response['choices'][0]['message']['content']
            print(f"\n{'='*20} 🤖 AI 回答 {'='*20}")
            print(answer)
            print(f"{'='*50}\n")
            return answer
        else:
            msg = "LLM 未返回有效结果。"
            print(msg)
            return msg
            
    except Exception as e:
        msg = f"调用 LLM 失败: {e}"
        print(msg)
        return msg

if __name__ == "__main__":
    # 1. 构建库 (如果第一次运行)
    # 注意：如果只想测试查询，可以注释掉 build_vector_db
    # build_vector_db()
    
    # 2. RAG 测试
    print("\n" + "="*50)
    question = "What is Tesla's total revenue in 2023?"
    rag_chat(question)
    
    print("\n" + "="*50)
    question2 = "What are the risk factors mentioned?"
    rag_chat(question2)
