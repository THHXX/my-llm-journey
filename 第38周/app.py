import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_engine import SmartCSBot

# 初始化 FastAPI 应用
app = FastAPI(title="RAG 智能客服 API", description="基于 FastAPI 和 LangChain 的 RAG 问答服务")

# 初始化 RAG 机器人
# 注意：这里需要确保 DASHSCOPE_API_KEY 环境变量已经设置
if "DASHSCOPE_API_KEY" not in os.environ:
    print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY 环境变量，请确保已配置！")

# 实例化机器人，指定数据目录
# 假设我们在第38周目录下运行，数据目录就在 ./data
qa_bot = SmartCSBot(data_dir="./data")

# 定义请求体模型
class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG API! Use POST /ask to chat."}

@app.post("/ask")
def ask(query: Query):
    if not query.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # 调用 RAG 引擎的 chat 方法
    answer = qa_bot.chat(query.question)
    
    return {"answer": answer}
