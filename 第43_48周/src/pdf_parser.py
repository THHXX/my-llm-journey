import pdfplumber
import pandas as pd
import os

def extract_text_with_tables(pdf_path):
    """
    智能提取文本：将表格转换为 Markdown 格式嵌入文本中，保持上下文连贯
    :param pdf_path: PDF 文件路径
    :return: 包含 Markdown 表格的完整文本
    """
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}")
        return ""
    
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"🔍 正在智能解析 {pdf_path} (表格转Markdown)...")
            for i, page in enumerate(pdf.pages):
                # 1. 提取纯文本
                text = page.extract_text() or ""
                
                # 2. 提取表格并转换为 Markdown
                tables = page.extract_tables()
                table_mds = []
                if tables:
                    for table in tables:
                        # 过滤无效表格 (行数太少或空内容)
                        if not table or len(table) < 2:
                            continue
                            
                        # 清洗 None 值
                        clean_table = [[str(cell).replace('\n', ' ') if cell else "" for cell in row] for row in table]
                        
                        try:
                            # 转换为 DataFrame 再转 Markdown
                            df = pd.DataFrame(clean_table[1:], columns=clean_table[0])
                            # 移除空列
                            df = df.dropna(axis=1, how='all')
                            if not df.empty:
                                md = df.to_markdown(index=False)
                                table_mds.append(f"\n\n[Table from Page {i+1}]\n{md}\n\n")
                        except Exception:
                            continue

                # 3. 拼接策略：简单将表格追加在每页文本之后
                # (更高级的策略是根据坐标插入回原文位置，但实现较复杂)
                page_content = text + "".join(table_mds)
                full_text += f"\n--- Page {i+1} ---\n{page_content}"
                
        return full_text
    except Exception as e:
        print(f"解析出错: {e}")
        return ""

# 保留旧接口兼容性
extract_text_from_pdf = extract_text_with_tables

def extract_text_with_page_infos(pdf_path):
    """
    【极致优化版】提取文本并保留页码信息
    :return: List[Dict] -> [{"page": 1, "text": "..."}]
    """
    if not os.path.exists(pdf_path):
        return []
    
    pages_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"🔍 [Pro] 正在深度解析 {pdf_path} (带页码追踪)...")
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text() or ""
                
                # 表格处理逻辑 (同上)
                tables = page.extract_tables()
                table_mds = []
                if tables:
                    for table in tables:
                        if not table or len(table) < 2: continue
                        clean_table = [[str(cell).replace('\n', ' ') if cell else "" for cell in row] for row in table]
                        try:
                            df = pd.DataFrame(clean_table[1:], columns=clean_table[0])
                            df = df.dropna(axis=1, how='all')
                            if not df.empty:
                                md = df.to_markdown(index=False)
                                table_mds.append(f"\n\n[Table from Page {page_num}]\n{md}\n\n")
                        except: pass

                full_page_text = text + "".join(table_mds)
                # 标记页码边界，方便后续 debug，但主要依靠返回结构
                pages_data.append({
                    "page": page_num,
                    "text": full_page_text
                })
                
        return pages_data
    except Exception as e:
        print(f"解析出错: {e}")
        return []

def extract_tables_from_pdf(pdf_path):
    """
    提取 PDF 中的表格
    :param pdf_path: PDF 文件路径
    :return: 包含表格 DataFrame 的列表
    """
    if not os.path.exists(pdf_path):
        return []

    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    # 简单的表格处理：假设第一行是表头
                    # 实际情况可能更复杂，需要后续优化
                    if table and len(table) > 1:
                        # 清洗数据：移除 None 和空字符串
                        clean_table = [[cell if cell else "" for cell in row] for row in table]
                        df = pd.DataFrame(clean_table[1:], columns=clean_table[0])
                        all_tables.append({"page": i+1, "data": df})
        return all_tables
    except Exception as e:
        print(f"表格提取出错: {e}")
        return []

if __name__ == "__main__":
    # 获取当前脚本所在目录的上一级目录下的 data 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    # 自动查找 data 目录下的第一个 PDF 文件
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if pdf_files:
        pdf_file = os.path.join(data_dir, pdf_files[0])
        print(f"找到文件: {pdf_file}")
        
        # 1. 提取文本
        print("-" * 30)
        print("1. 开始提取文本...")
        text = extract_text_from_pdf(pdf_file)
        print(f"文本提取完成，长度: {len(text)} 字符")
        print("文本预览 (前500字符):\n", text[:500])
        
        # 2. 提取表格
        print("-" * 30)
        print("2. 开始提取表格...")
        tables = extract_tables_from_pdf(pdf_file)
        print(f"共提取到 {len(tables)} 个表格")
        
        if tables:
            print("第一个表格预览:")
            print(tables[0]['data'].head())
            
    else:
        print(f"请将财报 PDF 文件放入以下目录: {data_dir}")
