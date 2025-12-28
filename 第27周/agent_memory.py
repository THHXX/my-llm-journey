import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.llms import Tongyi
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- 1. 配置 ---
# 确保 DASHSCOPE_API_KEY 环境变量已设置
# os.environ["DASHSCOPE_API_KEY"] = "sk-..."

print(f"{'='*20} 初始化 Agent (带记忆) {'='*20}")

# --- 2. 定义工具 ---
# 为了演示记忆功能，我们定义一个简单的模拟天气工具
def get_weather(location):
    return f"{location} 今天晴朗，气温 25 度。"

tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="查询天气时使用。输入为城市名称。"
    )
]

# --- 3. 定义带记忆的 Prompt ---
# 关键点：我们在 Prompt 中必须显式加入 {chat_history} 占位符
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# --- 4. 初始化 LLM 和 Memory ---
llm = Tongyi(model="qwen-max")

# memory_key="chat_history" 必须与 Prompt 中的 {chat_history} 对应
memory = ConversationBufferMemory(memory_key="chat_history")

# --- 5. 创建 Agent 执行器 ---
agent = create_react_agent(llm, tools, prompt)

# 将 memory 传入 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True,
    handle_parsing_errors=True
)

# --- 6. 运行多轮对话 ---
print("\n🤖 [第一轮] 告诉 Agent 我的名字...")
agent_executor.invoke({"input": "你好，我是小明。"})

print("\n🤖 [第二轮] 询问 Agent 是否记得我...")
# 这里我们并没有在 input 中提名字，Agent 必须从 memory 中获取
response = agent_executor.invoke({"input": "我刚才说我叫什么名字？"})

print(f"\n✅ 最终回复: {response['output']}")

print("\n🤖 [第三轮] 结合工具使用...")
agent_executor.invoke({"input": "我现在在北京，这里天气怎么样？"})