import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class SmartCSBotMilvus:
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
        """初始化或加载向量数据库 (Milvus)"""
        print("📦 正在连接 Milvus 并加载知识库...")
        
        # 1. 检查数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"⚠️ 警告: 数据目录 {self.data_dir} 不存在，已创建。请放入 txt 文件。")
            # 即使没有文件，我们也尝试连接 Milvus，可能之前已经存过数据
        
        # 2. 加载目录下所有 txt 文件
        loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs = loader.load()
        
        # 3. 配置 Embedding
        embeddings = DashScopeEmbeddings(model="text-embedding-v1")

        if docs:
            print(f"📄 发现 {len(docs)} 个文档，正在处理...")
            # 切分文档
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(docs)
            
            # 创建/更新 Milvus 集合
            # Milvus 会自动处理增量插入 (但在生产环境需要更复杂的去重逻辑)
            # 这里简单演示：如果有新文档就插入
            self.vectorstore = Milvus.from_documents(
                texts,
                embeddings,
                collection_name="rag_knowledge_base",
                connection_args={"host": "localhost", "port": "19530"}
            )
            print(f"✅ 已向 Milvus 插入 {len(texts)} 个文档块。")
        else:
            print("⚠️ 本地 data 目录为空，尝试直接连接现有的 Milvus 集合...")
            # 如果本地没文件，尝试直接连接
            try:
                self.vectorstore = Milvus(
                    embedding_function=embeddings,
                    collection_name="rag_knowledge_base",
                    connection_args={"host": "localhost", "port": "19530"}
                )
                print("✅ 成功连接到现有 Milvus 集合。")
            except Exception as e:
                print(f"❌ 连接 Milvus 失败: {e}")
                return

        # 创建 QA 链
        self.init_qa_chain()
        print("✅ RAG 系统初始化完成！")

    def init_qa_chain(self):
        """初始化问答链"""
        if not self.vectorstore:
            return
            
        llm = ChatTongyi(model=self.model_name)
        
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
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

    def chat(self, query):
        """对外提供的对话接口"""
        if not self.qa_chain:
            return "⚠️ 系统未就绪：请确保 Milvus 正在运行且已存入数据。"
            
        try:
            result = self.qa_chain.invoke({"question": query})
            return result['answer']
        except Exception as e:
            return f"❌ 发生错误: {str(e)}"

    def clear_memory(self):
        self.memory.clear()

if __name__ == "__main__":
    # 测试代码
    print("🚀 启动 Milvus RAG 测试...")
    bot = SmartCSBotMilvus(data_dir="./data")
    
    # 模拟对话
    questions = ["你们支持退货吗？", "多少钱包邮？"]
    for q in questions:
        print(f"\nUser: {q}")
        ans = bot.chat(q)
        print(f"Bot: {ans}")
