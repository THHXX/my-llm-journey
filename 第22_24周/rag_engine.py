import json
import numpy as np
from sentence_transformers import SentenceTransformer

class LawRAG:
    def __init__(self, json_path):
        print("📥 正在加载知识库...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.knowledge = json.load(f)
        
        # 提取所有文本
        self.corpus = [item['content'] for item in self.knowledge]
        
        print("🧠 正在加载检索模型 (moka-ai/m3e-small)...")
        # 使用中文效果很好的轻量级嵌入模型
        self.encoder = SentenceTransformer(r'C:\Users\JYJYJ\.cache\huggingface\hub\models--moka-ai--m3e-small\snapshots\44c696631b2a8c200220aaaad5f987f096e986df')
        
        print("⚡ 正在构建向量索引...")
        self.corpus_embeddings = self.encoder.encode(self.corpus)

    def search(self, query, top_k=2):
        # 1. 把用户的问题变成向量
        query_embedding = self.encoder.encode(query)
        
        # 2. 计算相似度 (Cosine Similarity)
        # 简单的点积计算
        similarities = np.dot(self.corpus_embeddings, query_embedding)
        
        # 3. 找出最相似的 top_k 个
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(self.corpus[idx])
            
        return results
