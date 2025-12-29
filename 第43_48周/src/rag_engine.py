import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_parser import extract_text_from_pdf, extract_text_with_page_infos
from llm_interface import LocalLLM, QwenCloudLLM
import datetime

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

    # 1. 提取文本 (带页码)
    print(f"1. 正在读取 {pdf_filename} ...")
    # full_text = extract_text_from_pdf(pdf_path) # 旧接口
    pages_data = extract_text_with_page_infos(pdf_path) # 新接口: [{"page": 1, "text": "..."}, ...]
    
    if not pages_data:
        print("提取文本失败。")
        return None
    print(f"   提取成功，共 {len(pages_data)} 页。")

    # 2. 文本切分 (Chunking) - 按页切分以保留页码
    print("2. 正在进行文本切分 (Page-wise Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 稍微减小一点，因为是单页切分
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    chunk_counter = 0
    for page_info in pages_data:
        page_num = page_info['page']
        page_text = page_info['text']
        
        page_chunks = text_splitter.split_text(page_text)
        
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": pdf_filename,
                "page": page_num, # 关键：存入页码！
                "chunk_index": chunk_counter
            })
            all_ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1
            
    print(f"   切分完成，共生成 {len(all_chunks)} 个文本块。")

    # 初始化 ChromaDB 客户端
    print(f"3. 初始化 ChromaDB (持久化路径: {CHROMA_DB_DIR})...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # -----------------------------------------------------------
    # 使用 ModelScope (阿里云) 下载模型，解决国内网络问题
    # -----------------------------------------------------------
    # 【重要】设置缓存目录到 D 盘，避免占用 C 盘空间
    os.environ['MODELSCOPE_CACHE'] = 'D:\\ModelScope_Cache'
    
    # 🌟 升级为 BGE-M3 (中文/多语言检索最强)
    # 虽然模型稍大 (约 2GB)，但效果质变，支持中英混合检索
    model_id = "Xorbits/bge-m3" 
    model_name_or_path = "BAAI/bge-m3" # fallback name

    try:
        from modelscope import snapshot_download
        print("   🚀 正在使用 ModelScope (阿里云) 下载 BGE-M3 模型...")
        print(f"   📂 缓存目录: {os.environ['MODELSCOPE_CACHE']}")
        model_dir = snapshot_download(model_id, revision='master')
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
    
    # ChromaDB 建议分批插入，防止一次性过大
    batch_size = 100
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        print(f"   正在处理批次 {i//batch_size + 1}/{total_batches}...")
        collection.add(
            documents=all_chunks[i:batch_end],
            metadatas=all_metadatas[i:batch_end],
            ids=all_ids[i:batch_end]
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
    model_id = "Xorbits/bge-m3" 
    model_name_or_path = "BAAI/bge-m3"

    try:
        from modelscope import snapshot_download
        # 此时应该已经缓存了，不会重复下载
        model_dir = snapshot_download(model_id, revision='master')
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
        return [], []
    
    return results['documents'][0], results['metadatas'][0]

def rag_chat(query_text, collection_name="financial_reports", llm_type="local"):
    """
    RAG 对话：检索 -> 生成
    :param llm_type: 'local' (本地LLM) 或 'cloud' (阿里云Qwen)
    """
    # 1. 检索相关文档
    # 💡 策略优化：Cloud 模型 (Qwen-Max) 上下文窗口很大，可以检索更多文档以提高召回率
    # Local 模型上下文有限，保持较少的检索数量
    target_n_results = 10 if llm_type == "cloud" else 3
    
    retrieved_docs, metadatas = query_vector_db(query_text, collection_name, n_results=target_n_results)
    
    if not retrieved_docs:
        print("未找到相关文档，无法回答。")
        return "未找到相关文档，无法回答。" # Return string for UI

    # 2. 构建 Prompt (针对小模型优化：中文指令，强制简短)
    context = "\n\n".join(retrieved_docs)
    
    system_prompt = """你是一个专业的金融分析师。请根据提供的【上下文】回答用户的【问题】。

规则：
1. 上下文可能是英文的，请你理解后用**中文**回答。
2. 必须完全基于【上下文】中的信息回答，不要使用你自己的外部知识。
3. 如果【上下文】中没有答案，请直接回答“根据提供的文档，无法找到相关信息”。
4. 严禁编造数字或事实。
5. 【重要】注意数字单位！如果文中单位是 "in millions" (百万)，请在回答中明确指出 (例如：$78,509 million 或 785.09亿美元)。
6. 保持回答简洁明了，直接给出结论或数字。
7. 回答结束后，请务必输出 "[END]" 并停止。

【进阶功能 - 图表生成协议】：（能够生成图表就一定要有json）
如果用户的问题涉及**数据对比**或**趋势分析**，且上下文中包含足够的数据，请在回答的最后（[END]之前）附带一个 JSON 代码块，用于生成图表。
格式如下：
```json
{
    "type": "bar",  // 图表类型: "bar" (柱状图), "line" (折线图)
    "title": "图表标题",
    "data": {
        "x": ["2021", "2022", "2023"], // X轴标签
        "y": [100, 200, 300],          // Y轴数值 (纯数字，不要带单位)
        "x_label": "年份",             // X轴名称
        "y_label": "收入 (百万美元)"    // Y轴名称
    }
}
```
注意：
- 仅在数据充足时生成。
- JSON 必须包裹在 ```json 和 ``` 之间。
- Y轴数据必须是纯数字。
"""
    
    user_prompt = f"""
【上下文】：
{context}

【问题】：
{query_text}

【回答】：
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 3. 调用 LLM
    print(f"\n🤖 正在请求 LLM ({llm_type}) 生成回答...")
    
    # 收集来源信息
    sources_text = "\n\n**📚 参考来源:**\n"
    seen_pages = set()
    for i, doc in enumerate(retrieved_docs):
        # 重新查询 metadata (因为 query_vector_db 只返回了 text)
        # 这里为了性能，我们假设 query_vector_db 内部打印了 metadata，
        # 但为了在 UI 展示，最好让 query_vector_db 返回完整对象。
        # 简化处理：我们在 query_vector_db 里直接修改返回结构，或者在这里简单附加说明。
        pass
    
    # --- 临时修正：为了获取 metadata，我们需要修改 query_vector_db 的返回签名 ---
    # 但为了不破坏现有逻辑，我们先把 sources_text 留空，
    # 更好的做法是修改 query_vector_db 返回 (docs, metadatas)
    
    try:
        if llm_type == "cloud":
            llm = QwenCloudLLM()
            # 云端模型能力更强，temperature 可以稍微高一点点，或者保持低位以求稳
            response = llm.chat(
                messages,
                temperature=0.2, # 稍微给一点灵活性，但保持严谨
            )
        else:
            llm = LocalLLM() # 会自动确保服务运行
            # 降低 temperature 到 0.1 以减少幻觉
            # 增加 stop 停止词，防止模型自问自答
            # 增加 frequency_penalty 防止重复循环 (如 "鲁鲁鲁...")
            response = llm.chat(
                messages, 
                temperature=0.1, 
                stop=["[END]", "【问题】", "User:", "Question:", "\n\n\n"],
                frequency_penalty=1.2 
            ) 
        
        if response:
            answer = response['choices'][0]['message']['content']
            # 清理可能的 [END] 标记
            answer = answer.replace("[END]", "").strip()
            
            print(f"\n{'='*20} 🤖 AI 回答 ({llm_type}) {'='*20}")
            print(answer)
            print(f"{'='*50}\n")
            
            # --- 极致优化：追加引用来源 ---
            # 去重：同一页可能被切分成多个 chunk，我们只显示唯一的页码
            unique_sources = sorted(list(set([m.get('page', '?') for m in metadatas])))
            
            citation_str = "\n\n---\n**📚 参考来源:**\n"
            # 如果来源太多，只显示前5个页码，避免太长
            if len(unique_sources) > 5:
                pages_str = ", ".join([str(p) for p in unique_sources[:5]]) + "..."
            else:
                pages_str = ", ".join([str(p) for p in unique_sources])
                
            source_file = metadatas[0].get('source', 'Unknown')
            citation_str += f"- 文件: {source_file}\n- 页码: {pages_str}\n"
            
            final_output = answer + citation_str
            # -----------------------------------------------

            # --- 新增：自动保存对话日志 (方便 AI 助手读取) ---
            try:
                log_path = os.path.join(os.path.dirname(CURRENT_DIR), "chat_history.log")
                with open(log_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n--- [{timestamp}] Type: {llm_type} ---\n")
                    f.write(f"Q: {query_text}\n")
                    f.write(f"A: {answer}\n")
                    f.write(f"Sources: Page {pages_str}\n")
                    f.write("-" * 30 + "\n")
            except Exception as log_err:
                print(f"写入日志失败: {log_err}")
            # -----------------------------------------------

            return final_output
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
    # 使用中文提问，测试模型是否能用中文回答
    question = "Tesla 2023年的总收入是多少？" 
    rag_chat(question)
    
    print("\n" + "="*50)
    question2 = "财报中提到了哪些风险因素？"
    rag_chat(question2)
