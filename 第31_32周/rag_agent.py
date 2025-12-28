import os

# 1. 禁用代理，防止国内 API 访问失败
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

print("🚀 正在初始化 RAG + Agent 系统...")

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ==========================================
# 第一步：RAG 系统构建 (知识库)
# ==========================================

# 1. 加载本地法律文档
doc_path = os.path.join(os.path.dirname(__file__), "law_sample.txt")
if not os.path.exists(doc_path):
    print(f"❌ 错误：找不到文件 {doc_path}")
    exit(1)

loader = TextLoader(doc_path, encoding='utf-8')
docs = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

# 3. 创建向量数据库 (使用 DashScope Embedding，无需本地下载模型)
# 注意：DashScopeEmbeddings 需要 DASHSCOPE_API_KEY
print("📦 正在构建向量数据库 (Embedding)...")
embeddings = DashScopeEmbeddings(model="text-embedding-v1")
vectorstore = FAISS.from_documents(texts, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# ==========================================
# 第二步：Agent 系统构建 (大脑 + 工具)
# ==========================================

# 1. 定义工具：将检索器封装为 Agent 可用的工具
tool = create_retriever_tool(
    retriever,
    "search_legal_docs",
    "用于搜索法律法规和合同相关知识。当用户询问法律问题时，必须使用此工具查找信息。"
)
tools = [tool]

# 2. 初始化大模型 (Qwen-Max)
llm = ChatTongyi(model="qwen-max")

# 3. 定义 Prompt 模板 (ReAct 模式)
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

# 4. 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 5. 创建 Agent 执行器 (带记忆功能)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # 打印思考过程
    handle_parsing_errors=True # 容错处理
)

# ==========================================
# 第三步：运行测试
# ==========================================

def ask(question):
    print(f"\n🟢 用户提问: {question}")
    try:
        result = agent_executor.invoke({"input": question})
        print(f"🤖 AI 回答: {result['output']}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    # 测试问题 1：需要检索文档
    ask("合同违约需要赔偿什么？")
    
    # 测试问题 2：基于上下文的多轮对话 (记忆测试)
    ask("那如果违约金定得太高怎么办？")
