import streamlit as st
import os
import sys
import re
import json
import pandas as pd
import plotly.express as px

# 将 src 目录添加到 sys.path，以便导入 rag_engine
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from rag_engine import rag_chat, get_embedding_function

# -----------------------------------------------------------------------------
# 🚀 性能优化：缓存 Embedding 模型
# -----------------------------------------------------------------------------
# st.cache_resource 专门用于缓存全局资源 (如数据库连接、ML模型)
# 这样，模型只需加载一次，后续所有用户请求都可以复用，极大提升响应速度！
@st.cache_resource
def load_cached_embedding_function():
    return get_embedding_function()

# 预加载模型 (应用启动时执行一次)
embedding_function = load_cached_embedding_function()

def render_chart_from_response(response_text):
    """
    从回答中提取 JSON 并绘制图表
    """
    # 正则匹配 ```json ... ```
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            chart_data = json.loads(json_match.group(1))
            
            if "type" in chart_data and "data" in chart_data:
                st.markdown(f"### 📊 {chart_data.get('title', '数据可视化')}")
                
                # 构建 DataFrame
                df = pd.DataFrame({
                    chart_data['data'].get('x_label', 'X'): chart_data['data']['x'],
                    chart_data['data'].get('y_label', 'Y'): chart_data['data']['y']
                })
                # 设置索引以便 Streamlit 自动识别 X 轴
                # df = df.set_index(chart_data['data'].get('x_label', 'X'))
                
                x_col = chart_data['data'].get('x_label', 'X')
                y_col = chart_data['data'].get('y_label', 'Y')

                if chart_data['type'] == 'bar':
                    # st.bar_chart(df)
                    fig = px.bar(df, x=x_col, y=y_col, title=chart_data.get('title', ''), text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_data['type'] == 'line':
                    # st.line_chart(df)
                    fig = px.line(df, x=x_col, y=y_col, title=chart_data.get('title', ''), markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"暂不支持的图表类型: {chart_data['type']}")
                    
        except json.JSONDecodeError:
            pass # JSON 解析失败，忽略
        except Exception as e:
            st.warning(f"图表渲染失败: {e}")

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

    # --- LLM 选择 (环境感知) ---
    # 检查是否在 Streamlit Cloud 上运行 (通过检查环境变量)
    # Streamlit Cloud 会设置 STREAMLIT_SHARING_MODE = "true"
    IS_ON_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "true"

    if IS_ON_STREAMLIT_CLOUD:
        # 在云端，强制使用 Cloud 模式并禁用选项
        st.info("☁️ 云端环境，仅支持在线模型。")
        llm_option = "Cloud (阿里云 Qwen-Max)"
        st.radio(
            "🤖 模型选择",
            (llm_option,),
            index=0,
            disabled=True,
            help="云端部署环境不支持本地模型。"
        )
    else:
        # 在本地，提供两种选择
        llm_option = st.radio(
            "🤖 模型选择",
            ("Local (本地 Qwen1.5-0.5B)", "Cloud (阿里云 Qwen-Max)"),
            help="本地模式完全离线；云端模式需要 API Key，能力更强。"
        )
    
    llm_type = "local" if "Local" in llm_option else "cloud"

    if llm_type == "cloud":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        key_file_exists = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "key.txt"))
        
        if api_key:
            st.success("✅ 已检测到环境变量 DASHSCOPE_API_KEY")
        elif key_file_exists:
            st.success("✅ 已检测到 key.txt 文件")
        else:
            st.warning("⚠️ 未检测到 API Key！请设置环境变量 DASHSCOPE_API_KEY 或在项目根目录创建 key.txt。")
    
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
        # 如果是 AI 的回答，尝试渲染图表
        if message["role"] == "assistant":
            render_chart_from_response(message["content"])

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
                # 传入缓存的 embedding_function，避免重复加载模型
                response = rag_chat(
                    query_text=prompt,
                    collection_name=collection_name,
                    llm_type=llm_type,
                    embedding_function=embedding_function
                )
                
                # 显示回答
                st.markdown(response)
                
                # --- 新增：尝试解析并绘制图表 ---
                render_chart_from_response(response)
                # ------------------------------------
                
                # 保存回答到历史
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"发生错误: {e}")
