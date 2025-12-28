import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class SmartCSBot:
    def __init__(self, data_dir="./data", model_name="qwen-max"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question", 
            output_key="answer"
        )
        
        # 强制禁用代理
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        
        # 初始化知识库
        self.init_vectorstore()
        
    def init_vectorstore(self):
        """初始化或加载向量数据库"""
        print("📦 正在加载知识库...")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"⚠️ 警告: 数据目录 {self.data_dir} 不存在，已创建。请放入 txt 文件。")
            return

        # 加载目录下所有 txt 文件
        loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs = loader.load()
        
        if not docs:
            print("⚠️ 警告: 知识库为空，请在 data 目录下放入 txt 文档。")
            return

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        
        # 创建向量库
        embeddings = DashScopeEmbeddings(model="text-embedding-v1")
        self.vectorstore = FAISS.from_documents(texts, embeddings)
        
        # 创建 QA 链
        self.init_qa_chain()
        print("✅ 知识库加载完成！")

    def init_qa_chain(self):
        """初始化问答链"""
        if not self.vectorstore:
            return
            
        llm = ChatTongyi(model=self.model_name)
        
        # 自定义 Prompt，赋予角色
        # 注意：使用 ConversationalRetrievalChain 时，chat_history 主要用于生成独立问题
        # 这里是回答问题的 Prompt，只需要 context 和 question
        prompt_template = """你是一个专业的电商智能客服“小蜜”。
请根据以下上下文（Context）回答用户的问题。
如果你不知道答案，请礼貌地回答“抱歉，暂时没有相关信息，请联系人工客服”，不要编造答案。
回答要亲切、自然，可以使用 emoji。

上下文：
{context}

用户提问：{question}
客服回答："""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # 使用 ConversationalRetrievalChain，它是处理多轮对话 RAG 的标准方式
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def chat(self, query):
        """对外提供的对话接口"""
        if not self.qa_chain:
            return "⚠️ 系统未就绪：知识库为空或初始化失败。"
            
        try:
            # 使用 invoke 而不是 run，兼容新版 LangChain
            # ConversationalRetrievalChain 接受 question
            result = self.qa_chain.invoke({"question": query})
            return result['answer']
        except Exception as e:
            return f"❌ 发生错误: {str(e)}"

    def clear_memory(self):
        self.memory.clear()
