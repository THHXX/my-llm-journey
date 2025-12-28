import time
import random
import json
import hashlib
from redis import Redis

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
# 模拟 API 调用的单次成本 (假设 $0.002 / 1k tokens)
COST_PER_CALL = 0.002 

# 连接 Redis (请确保在 WSL 中已启动 redis-server)
# host='localhost', port=6379, db=0 是默认配置
try:
    redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping() # 测试连接
    print("✅ Redis 连接成功！")
except Exception as e:
    print(f"❌ Redis 连接失败: {e}")
    print("💡 请检查：\n1. 是否在 WSL 中安装了 Redis (sudo apt install redis-server)\n2. 是否启动了服务 (sudo service redis-server start)")
    exit()

# -----------------------------------------------------------------------------
# 2. 模拟耗时且昂贵的 API 调用
# -----------------------------------------------------------------------------
def mock_expensive_api_call(prompt):
    """
    模拟一个调用大模型的函数。
    它很慢 (sleep)，而且很贵 (计费)。
    """
    print(f"   [API] 正在请求云端模型: '{prompt}' ...")
    time.sleep(1.5)  # 模拟网络延迟和推理时间
    
    # 模拟返回结果
    response = f"这是针对问题 '{prompt}' 的智能回答 (由 AI 生成)"
    return response

# -----------------------------------------------------------------------------
# 3. 核心功能：带缓存的调用函数
# -----------------------------------------------------------------------------
def smart_query(prompt):
    """
    智能查询：先查缓存，没有再查 API
    """
    # 1. 生成缓存 Key (用 MD5 保证唯一性，防止 Key 过长)
    # 例如: "cache:qwen:e10adc3949ba59abbe56e057f20f883e"
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_key = f"cache:qwen:{prompt_hash}"

    # 2. 尝试从 Redis 读取
    cached_result = redis_client.get(cache_key)

    if cached_result:
        print(f"   [Cache] ✅ 命中缓存！直接返回结果 (省钱了！)")
        return cached_result, 0.0  # 成本为 0

    # 3. 如果缓存没命中，调用真实 API
    print(f"   [Cache] ❌ 未命中，必须调用 API...")
    result = mock_expensive_api_call(prompt)

    # 4. 写入 Redis (设置过期时间 1 小时 = 3600 秒)
    redis_client.set(cache_key, result, ex=3600)
    
    return result, COST_PER_CALL

# -----------------------------------------------------------------------------
# 4. 主程序：测试效果
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 清空之前的测试数据 (可选)
    # redis_client.flushdb()

    test_questions = [
        "如何学习 Python?",
        "什么是量化交易?",
        "如何学习 Python?",  # 重复问题，应该命中缓存
        "WSL 是什么?",
        "什么是量化交易?",  # 重复问题，应该命中缓存
    ]

    total_cost = 0.0
    start_time = time.time()

    print("-" * 50)
    print("🚀 开始 API 成本优化测试")
    print("-" * 50)

    for i, q in enumerate(test_questions):
        print(f"\n📝 问题 {i+1}: {q}")
        
        # 调用智能查询
        answer, cost = smart_query(q)
        
        total_cost += cost
        print(f"   💰 本次成本: ${cost}")

    end_time = time.time()
    
    # 统计结果
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(f"💵 总成本: ${total_cost:.4f}")
    
    # 计算如果没有缓存的成本
    theoretical_cost = len(test_questions) * COST_PER_CALL
    saved_money = theoretical_cost - total_cost
    saved_percent = (saved_money / theoretical_cost) * 100 if theoretical_cost > 0 else 0

    print(f"📉 理论成本 (无缓存): ${theoretical_cost:.4f}")
    print(f"🛡️  节省金额: ${saved_money:.4f} (节省了 {saved_percent:.1f}%)")
    print("=" * 50)
