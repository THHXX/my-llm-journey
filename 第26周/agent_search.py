from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Tongyi
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# --- 1. 配置 ---
# 请设置您的通义千问 API Key
# os.environ["DASHSCOPE_API_KEY"] = "sk-..."

# 检查网络环境 (DuckDuckGo 需要科学上网)
def check_network():
    print("[系统日志] 正在检查网络连接...")
    try:
        # 尝试实例化 wrapper 并简单测试
        # backend="text" 是新版 duckduckgo-search 推荐的后端，且通常比 API 模式更稳定
        wrapper = DuckDuckGoSearchAPIWrapper(backend="text")
        # 简单测试一下
        wrapper.run("test")
        print("[系统日志] ✅ 网络通畅，DuckDuckGo 可用")
        return wrapper
    except Exception as e:
        print(f"[系统日志] ⚠️ 网络检测失败: {e}")
        print("[系统建议] 您的 VPN 可能未开启全局代理，或者节点被 DuckDuckGo 屏蔽。")
        print("[系统建议] 推荐使用 agent_search_serpapi.py (支持国内百度搜索)")
        return None

# --- 2. 定义工具 (Tools) ---
wrapper = check_network()

if wrapper:
    search_func = wrapper.run
else:
    # 失败时提供一个伪函数，避免程序直接崩溃
    def search_func(query):
        return "搜索失败：请检查网络或使用 agent_search_serpapi.py"

search_tool = Tool(
    name="WebSearch",
    func=search_func,
    description="当需要查询实时信息、新闻或不知道答案时使用此工具。输入应为具体的搜索关键词。"
)

def get_weather(location):
    print(f"\n[系统日志] 正在查询 {location} 的天气...")
    if "北京" in location:
        return "北京今天晴朗，气温 15-25 度，适合出行。"
    elif "上海" in location:
        return "上海今天有小雨，气温 18-22 度，出门请带伞。"
    else:
        return f"{location} 的天气数据暂时无法获取。"

custom_weather_tool = Tool(
    name="WeatherQuery",
    func=get_weather,
    description="当用户询问天气时使用此工具。输入应为城市名称。"
)

tools = [search_tool, custom_weather_tool]

# --- 3. 初始化 Agent ---
llm = Tongyi(model="qwen-max")

# 获取标准的 ReAct Prompt
# 如果 hub 拉取失败，我们可以手动定义
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

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# 使用 create_react_agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. 运行 ---
if __name__ == "__main__":
    questions = [
        "今天是几号？", 
        "第一性原理是什么？"
    ]
    
    for q in questions:
        print(f"\n{'='*20}\n🤖 用户提问: {q}")
        try:
            result = agent_executor.invoke({"input": q})
            print(f"✅ 最终答案: {result['output']}")
        except Exception as e:
            print(f"❌ 运行出错: {e}")