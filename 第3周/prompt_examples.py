"""
第3周：提示工程实战示例

运行命令（请在命令后补充中文注释以便记忆）:
python prompt_examples.py  # 运行全部示例，生成对比输出

环境要求：
- 安装 openai 新版 SDK: pip install openai  # 安装依赖
- 设置 OPENAI_API_KEY，必要时设置 OPENAI_BASE_URL 指向兼容的对话模型服务。
"""

from __future__ import annotations

import os
from typing import List, Dict

from openai import OpenAI


# 统一的演示素材：一段关于智能手表的用户评论，方便横向对比。
SAMPLE_REVIEW = """
我买了这款新发布的智能手表，外观比上一代更轻薄，屏幕更亮，
支持离线音乐和 NFC 支付，跑步 GPS 也比较准。遗憾是重度使用时
一天一充，语音助手有时反应慢。整体来说，日常通勤和运动都够用。
""".strip()


def get_client() -> OpenAI:
    """创建 OpenAI 客户端，便于后续切换不同基地址或密钥。"""
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.2) -> str:
    """
    统一的模型调用封装，便于后续替换模型或加上重试。
    messages: 符合 Chat Completions 的消息列表
    model: 不指定则使用环境变量 OPENAI_MODEL，否则退回 gpt-4o-mini
    temperature: 采样温度，越低越稳
    """
    client = get_client()
    chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=chosen_model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content


def run_summary_examples() -> None:
    """摘要生成：对比普通版与优化版 Prompt。"""
    print("\n=== 摘要生成 ===")

    # 普通版（你可自行修改，体会模糊提问的问题）
    prompt_basic = f"请用中文概括下面的评论：{SAMPLE_REVIEW}"

    # 优化版：设定角色、长度、结构化输出
    prompt_better = f"""
你是一名产品测评编辑，请用 50~80 字中文总结用户评论，输出两行要点：
1) 亮点：
2) 遗憾：
只基于提供的评论，不要加入外部推测。评论内容：{SAMPLE_REVIEW}
""".strip()

    messages_basic = [{"role": "user", "content": prompt_basic}]
    messages_better = [{"role": "system", "content": "你是一名严谨的科技编辑。"},
                       {"role": "user", "content": prompt_better}]

    print("\n-- Prompt A：普通版 --")
    print(prompt_basic)
    print("\n模型输出 A：")
    print(call_llm(messages_basic))

    print("\n-- Prompt B：优化版 --")
    print(prompt_better)
    print("\n模型输出 B：")
    print(call_llm(messages_better))


def run_sentiment_examples() -> None:
    """情感分析：对比简单问法与结构化 JSON 输出。"""
    print("\n=== 情感分析 ===")

    prompt_basic = f"判断这条评论的情感（正向/负向/中立）：{SAMPLE_REVIEW}"

    prompt_better = f"""
你是一名客服质检员。请对评论做情感分析，并以 JSON 输出：
{{
  "sentiment": "positive|neutral|negative",
  "confidence": 0.0-1.0,
  "reason": "简述理由"
}}
仅依据评论文本，禁止胡编信息。评论：{SAMPLE_REVIEW}
""".strip()

    messages_basic = [{"role": "user", "content": prompt_basic}]
    messages_better = [{"role": "system", "content": "你是一名严谨的客服质检员。"},
                       {"role": "user", "content": prompt_better}]

    print("\n-- Prompt A：普通版 --")
    print(prompt_basic)
    print("\n模型输出 A：")
    print(call_llm(messages_basic))

    print("\n-- Prompt B：优化版 --")
    print(prompt_better)
    print("\n模型输出 B：")
    print(call_llm(messages_better))


def run_qa_examples() -> None:
    """问答：对比随便问与限定知识来源。"""
    print("\n=== 问答 ===")

    question = "这款手表的续航如何？"
    prompt_basic = f"根据下列描述回答问题：{question}\n描述：{SAMPLE_REVIEW}"

    prompt_better = f"""
你只能依据提供的描述回答，如果描述里没有答案请回复“不知道”。
问题：{question}
描述：{SAMPLE_REVIEW}
输出格式：
- 若有答案，直接用中文简洁回答；
- 若无答案，只输出：不知道。
""".strip()

    messages_basic = [{"role": "user", "content": prompt_basic}]
    messages_better = [{"role": "system", "content": "你是一名只引用给定资料的回答助手。"},
                       {"role": "user", "content": prompt_better}]

    print("\n-- Prompt A：普通版 --")
    print(prompt_basic)
    print("\n模型输出 A：")
    print(call_llm(messages_basic))

    print("\n-- Prompt B：优化版 --")
    print(prompt_better)
    print("\n模型输出 B：")
    print(call_llm(messages_better))


def run_code_examples() -> None:
    """代码生成：对比模糊需求与精确规格。"""
    print("\n=== 代码生成 ===")

    prompt_basic = "写一段 Python 代码给我，能把列表排序。"

    prompt_better = """
请用 Python 实现函数 sort_scores(scores: list[int]) -> list[int]：
- 升序排序，保持整数类型
- 允许负数与重复值
- 不要使用内置 sort/sorted（练习算法表达）
- 代码后附上 1 个示例，演示输入 [3, -1, 3, 0] 的输出
""".strip()

    messages_basic = [{"role": "user", "content": prompt_basic}]
    messages_better = [{"role": "system", "content": "你是一名编写可读代码的工程师。"},
                       {"role": "user", "content": prompt_better}]

    print("\n-- Prompt A：普通版 --")
    print(prompt_basic)
    print("\n模型输出 A：")
    print(call_llm(messages_basic))

    print("\n-- Prompt B：优化版 --")
    print(prompt_better)
    print("\n模型输出 B：")
    print(call_llm(messages_better))


def run_dialogue_examples() -> None:
    """多轮对话引导：对比宽泛聊天与目标导向脚本。"""
    print("\n=== 多轮对话引导 ===")

    # 宽泛版：缺少目标约束
    messages_basic = [
        {"role": "system", "content": "你是聊天机器人。"},
        {"role": "user", "content": "你好，我们聊聊这款智能手表？"},
    ]

    # 目标导向版：设定对话目的与轮次规则
    messages_better = [
        {"role": "system", "content": "你是一名智能手表导购助理，目标是帮助用户决定是否购买。"},
        {"role": "user", "content": "请按以下步骤对话，不要跳步：\n"
                                    "1) 先问用户主要用途（通勤/运动/健康监测）\n"
                                    "2) 再问预算区间\n"
                                    "3) 再确认最看重的 2 个功能\n"
                                    "4) 根据用户回答给出是否推荐及 2 条理由，50 字内\n"
                                    "如果信息不足，请继续追问，不要编造。用户开场：我想看看这款手表"},
    ]

    print("\n-- Prompt A：普通版 --")
    for m in messages_basic:
        print(f"{m['role']}: {m['content']}")
    print("\n模型输出 A：")
    print(call_llm(messages_basic))

    print("\n-- Prompt B：优化版 --")
    for m in messages_better:
        print(f"{m['role']}: {m['content']}")
    print("\n模型输出 B（示例仅跑 1 轮，你可手动继续对话）：")
    print(call_llm(messages_better))


def main() -> None:
    """按顺序运行 5 个示例，输出对比结果。"""
    run_summary_examples()
    run_sentiment_examples()
    run_qa_examples()
    run_code_examples()
    run_dialogue_examples()
    print("\n提示：为方便截图，可单独注释/取消注释上面的函数调用。")


if __name__ == "__main__":
    main()

