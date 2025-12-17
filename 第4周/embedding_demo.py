"""
第4周：文本向量化与相似度计算示例

运行方式（命令后带中文注释便于记忆）：
python embedding_demo.py  # 计算两句中文的余弦相似度

依赖：
- pip install sentence-transformers scikit-learn  # 安装向量模型与相似度函数

模型：
- 默认使用 sentence-transformers/all-MiniLM-L6-v2（兼容中英文，体积小、下载快）
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main() -> None:
    """加载模型，生成两句中文的向量并计算余弦相似度。"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # 可自定义的两句话：一组相似例子
    sent1 = "我喜欢学习大模型"
    sent2 = "我很喜欢研究大语言模型"

    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)

    sim = cosine_similarity([emb1], [emb2])[0][0]

    print(f"句子1: {sent1}")
    print(f"句子2: {sent2}")
    print(f"余弦相似度: {sim:.4f}")
    print("提示：一般 >0.7 说明语义相近，可再尝试不相似的句子进行对比。")


if __name__ == "__main__":
    main()

