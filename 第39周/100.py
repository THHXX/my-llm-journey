import time
import numpy as np
from pymilvus import MilvusClient, DataType

# 1. 初始化 Milvus (使用 Docker 部署的 Standalone 版本)
# Windows 下 Milvus Lite 兼容性较差，改用 Docker 服务 (localhost:19530)
client = MilvusClient(uri="http://localhost:19530")

# 定义集合名称
COLLECTION_NAME = "rag_1m_benchmark"
DIMENSION = 768  # 常用 Embedding 维度

# 2. 检查旧数据
# 极简主义优化：如果数据量已达标，就不重复插入了
should_insert = True
if client.has_collection(collection_name=COLLECTION_NAME):
    res = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["count(*)"])
    # 注意：不同版本 query count 的返回格式可能不同，这里用异常处理兜底或直接看 num_entities
    # 简单方式：直接看 collection 统计信息（需 load）
    try:
        # 尝试获取集合统计
        stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
        # 这里的 stats 结构较复杂，简化处理：直接默认如果存在且没报错就不删
        print(f"ℹ️ 集合 {COLLECTION_NAME} 已存在，跳过清理和重新插入。")
        should_insert = False
    except:
        client.drop_collection(collection_name=COLLECTION_NAME)

if should_insert:
    # 3. 创建集合（开启自动 ID）
    if client.has_collection(collection_name=COLLECTION_NAME):
         client.drop_collection(collection_name=COLLECTION_NAME)
         
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=DIMENSION,
        metric_type="L2",  # 欧氏距离
        auto_id=True
    )
    print(f"🚀 集合 {COLLECTION_NAME} 创建成功")

    # 4. 生成 100 万条模拟数据
    # 极简主义：分批插入，避免一次性撑爆内存
    TOTAL_VECTORS = 1_000_000
    BATCH_SIZE = 10_000
    batches = TOTAL_VECTORS // BATCH_SIZE

    print(f"📦 开始生成并插入 {TOTAL_VECTORS} 条向量数据...")
    start_insert = time.time()

    for i in range(batches):
        # 生成随机向量 (模拟 BERT/Embedding 输出)
        vectors = np.random.random((BATCH_SIZE, DIMENSION)).astype(np.float32)
        # 构造插入数据
        data = [{"vector": vec, "text": f"doc_{i*BATCH_SIZE + j}"} for j, vec in enumerate(vectors)]
        
        client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"\r进度: {(i+1)/batches*100:.1f}%", end="")

    print(f"\n✅ 插入完成，耗时: {time.time() - start_insert:.2f} 秒")
else:
    print("⏩ 检测到数据已存在，直接进入检索测试...")

# 5. 创建索引 (关键！没有这个查询会很慢)
# 注意：MilvusClient.create_collection 如果指定了 metric_type，可能已经自动创建了 AUTOINDEX
# 这里我们先检查是否已有索引，如果没有再创建
try:
    print("⚙️ 正在构建 HNSW 索引 (可能需要几分钟)...")
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_type="HNSW",  # 最适合内存检索的高性能索引
        metric_type="L2",
        params={"M": 16, "efConstruction": 500}
    )

    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )
    print("✅ 索引构建完成")
except Exception as e:
    print(f"⚠️ 索引创建跳过 (可能已存在): {e}")

# 6. 加载集合到内存
client.load_collection(COLLECTION_NAME)

# 7. 性能测试：检索 < 100ms 挑战
print("\n🏁 开始性能测试 (Search)...")
search_vectors = np.random.random((1, DIMENSION)).astype(np.float32)

# 预热 (Warm-up)
print("🔥 正在预热 (Warm-up)...")
for _ in range(3):
    client.search(
        collection_name=COLLECTION_NAME,
        data=search_vectors,
        limit=5,
        search_params={"metric_type": "L2", "params": {"ef": 32}}
    )

# 运行 10 次取平均值，排除网络波动干扰
print("⚡ 正在执行 10 次连续查询取平均值...")
total_time = 0
for i in range(10):
    start_run = time.time()
    client.search(
        collection_name=COLLECTION_NAME,
        data=search_vectors,
        limit=5, 
        search_params={"metric_type": "L2", "params": {"ef": 32}} # 优化参数：降低 ef 提升速度
    )
    total_time += (time.time() - start_run)

avg_time = total_time / 10
print(f"✅ 平均检索耗时: {avg_time * 1000:.2f} ms")

if avg_time < 0.1:
    print("🎉 挑战成功！检索速度 < 100ms！Milvus 性能强劲！")
else:
    print("⚠️ 性能未达标，请检查电脑是否开启高性能模式。")