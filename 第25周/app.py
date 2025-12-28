import gradio as gr
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- 配置 ---
# 如果环境变量未设置，请在此处设置
# os.environ["DASHSCOPE_API_KEY"] = "sk-..."

def translate_text(text):
    """
    调用 LangChain 进行翻译的核心函数
    """
    if not text:
        return "请输入要翻译的内容。"
    
    try:
        # 1. 初始化 LLM
        # 使用 qwen-max 模型，效果更好
        llm = Tongyi(model="qwen-max")
        
        # 2. 定义 Prompt 模板
        # 这里我们可以稍微搞复杂一点，让它支持中译英
        prompt = PromptTemplate.from_template("请将以下中文内容翻译成地道的英文：\n{text}")
        
        # 3. 构建 Chain (LCEL 语法)
        chain = prompt | llm | StrOutputParser()
        
        # 4. 执行翻译
        result = chain.invoke({"text": text})
        return result.strip()
        
    except Exception as e:
        return f"❌ 发生错误: {str(e)}\n请检查您的 DASHSCOPE_API_KEY 是否已设置。"

# --- 构建 Gradio 界面 ---
with gr.Blocks(title="LangChain 翻译助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔗 LangChain + Qwen 智能翻译机")
    gr.Markdown("### 第25周实战：使用 LangChain 构建的大模型应用")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                label="输入中文", 
                placeholder="请输入您想翻译的句子...", 
                lines=5
            )
            submit_btn = gr.Button("🚀 开始翻译", variant="primary")
            
        with gr.Column():
            output_box = gr.Textbox(
                label="英文译文", 
                placeholder="翻译结果将显示在这里...", 
                lines=5,
                interactive=False
            )
            
    # 绑定事件
    submit_btn.click(
        fn=translate_text, 
        inputs=input_box, 
        outputs=output_box
    )
    
    # 添加一些示例
    gr.Examples(
        examples=[
            ["LangChain 让开发大模型应用变得非常简单。"],
            ["今天天气真不错，我想去公园散步。"],
            ["人工智能正在以惊人的速度改变世界。"]
        ],
        inputs=input_box
    )

if __name__ == "__main__":
    print("🚀 启动 Gradio 服务...")
    demo.launch()
