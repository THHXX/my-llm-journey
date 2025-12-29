import os
from rag_engine import build_vector_db

if __name__ == "__main__":
    print("🚀 开始重建向量数据库 (应用新的表格解析策略)...")
    collection = build_vector_db()
    if collection:
        print("\n✅ 数据库重建成功！")
        print("现在运行 python src/rag_engine.py 测试效果吧！")
    else:
        print("\n❌ 重建失败，请检查错误日志。")
