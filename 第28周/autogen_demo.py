import os
import autogen

# --- 1. 配置 LLM (核心修改) ---
# AutoGen 使用 OpenAI 兼容接口连接通义千问
config_list = [{
    "model": "qwen-max",
    "api_key": os.environ.get("DASHSCOPE_API_KEY"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "seed": 42, # 设置种子以保证结果可复现
}

print(f"{'='*20} 初始化多智能体团队 {'='*20}")

# --- 2. 定义角色 (Agents) ---

# 👩‍💼 产品经理：负责提需求
product_manager = autogen.AssistantAgent(
    name="Product_Manager",
    system_message="""你是产品经理。
    1. 你负责定义软件需求。
    2. 你需要清晰地描述我们要开发什么产品，包含哪些核心功能。
    3. 看到代码后，如果符合需求，请不要说话，让测试员去测。""",
    llm_config=llm_config
)

# 👨‍💻 程序员：负责写代码
programmer = autogen.AssistantAgent(
    name="Programmer",
    system_message="""你是程序员。
    1. 你根据产品经理的需求编写 Python 代码。
    2. 代码必须是完整的、可运行的，并且包含必要的注释。
    3. 不要使用伪代码，直接写出实现。""",
    llm_config=llm_config
)

# 🕵️ 测试员：负责测试和验收
tester = autogen.AssistantAgent(
    name="Tester",
    system_message="""你是测试员。
    1. 你负责检查程序员的代码。
    2. 如果代码有明显的逻辑错误或缺少功能，请提出具体的修改建议。
    3. 如果代码看起来完美且符合需求，请回复 'TERMINATE' 结束任务。""",
    llm_config=llm_config
)

# --- 3. 创建群聊 (GroupChat) ---
group_chat = autogen.GroupChat(
    agents=[product_manager, programmer, tester],
    messages=[],
    max_round=10  # 限制最大轮次，防止无限对话
)

# 创建群聊管理器 (主持人)
manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)

# --- 4. 发起任务 ---
print("🤖 任务启动：构建智能客服系统...")

# 由产品经理发起对话，直接抛出需求
product_manager.initiate_chat(
    manager,
    message="我们需要一个简单的智能客服系统（Python Class）。它应该包含：1. 添加知识库(问题-答案)的方法 2. 根据用户问题返回答案的方法（如果没有匹配的，返回默认回复）。请程序员编写代码。"
)