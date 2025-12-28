import streamlit as st
import os
import sys

# 将 src 目录添加到 sys.path，以便导入 rag_engine
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from rag_engine import rag_chat

# 页面配置
st.set_page_config(
    page_title="金融投研助手", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 金融投研助手 (RAG + LLM)")
st.markdown("---")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    collection_name = st.text_input(
        "📚 知识库集合", 
        value="financial_reports",
        help="ChromaDB 中的集合名称"
    )
    
    st.markdown("### 📊 功能说明")
    st.info(
        """
        本助手基于 **RAG (检索增强生成)** 技术：
        1. 检索本地财报 (PDF)
        2. 结合 LLM 生成专业回答
        3. 纯本地运行，数据不出域
        """
    )
    
    if st.button("🧹 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# --- 聊天逻辑 ---

# 1. 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 处理用户输入
if prompt := st.chat_input("请输入您的问题 (例如: Tesla 2023年的总收入是多少?)"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索财报并思考中..."):
            try:
                # 调用 RAG 引擎
                response = rag_chat(prompt, collection_name=collection_name)
                
                # 显示回答
                st.markdown(response)
                
                # 保存回答到历史
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"发生错误: {e}")
