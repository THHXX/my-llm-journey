import os
import gradio as gr
from rag_engine import SmartCSBot

# 0. 禁用代理与环境配置 (解决本地连接报错 net::ERR_ABORTED)
# 务必在导入其他库之前设置，确保不通过代理访问本地服务
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
# 移除可能存在的全局代理设置
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

# 1. 初始化智能客服 Agent
# 注意：这里我们使用全局实例。
# 在生产环境（如 ModelScope）中，如果是多用户并发，可能需要为每个 Session 创建实例。
# 但为了演示简单，我们暂用单例模式。
data_path = os.path.join(os.path.dirname(__file__), "data")
bot = SmartCSBot(data_dir=data_path)

def respond(message, history):
    """
    Gradio ChatInterface 的回调函数
    message: 用户当前输入
    history: 历史对话列表 [[user, bot], [user, bot]...]
    """
    if not message:
        return ""
        
    # 调用 RAG 引擎
    response = bot.chat(message)
    return response

# 2. 构建界面
# 使用 Soft 主题，更具亲和力
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
).set(
    body_background_fill="*neutral_50",
)

with gr.Blocks(title="极速达智能客服") as demo:
    gr.Markdown(
        """
        # 🤖 极速达物流 - 智能客服中心
        
        > 基于 RAG (检索增强生成) 技术，为您解答物流、售后、会员权益等问题。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            # 聊天窗口
            # Gradio 6.x 兼容性调整：移除已弃用的参数
            chat_interface = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=500, avatar_images=(None, "https://img.alicdn.com/imgextra/i4/O1CN01d2X6s51Jv1tQv1tQv_!!6000000001092-2-tps-200-200.png")),
                textbox=gr.Textbox(placeholder="请问有什么可以帮您？例如：退货运费谁出？", container=False, scale=7),
                title=None,
                description=None,
                # theme="soft", # Gradio 新版本中 ChatInterface 不支持直接传 theme，由外层 Blocks 控制
                examples=[
                    ["多少钱包邮？"],
                    ["怎么成为金卡会员？"],
                    ["收到货不喜欢能退吗？"],
                    ["客服几点下班？"]
                ],
                # retry_btn="🔄 重试", # Gradio 6.x 可能已移除或更改参数名
                # undo_btn="↩️ 撤回",
                # clear_btn="🗑️ 清空对话",
            )
        
        with gr.Column(scale=1):
            # 侧边栏信息
            gr.Markdown("### ℹ️ 系统状态")
            status_output = gr.Textbox(label="知识库状态", value="✅ 已加载", interactive=False)
            
            gr.Markdown("### 🛠️ 管理员操作")
            refresh_btn = gr.Button("🔄 重载知识库")
            
            def reload_kb():
                try:
                    bot.init_vectorstore()
                    return "✅ 重载成功"
                except Exception as e:
                    return f"❌ 失败: {str(e)}"
            
            refresh_btn.click(reload_kb, outputs=status_output)

            gr.Markdown(
                """
                ### 📝 使用说明
                1. 这是一个演示系统，数据来源于 `data/` 目录下的文档。
                2. 您可以询问左侧示例中的问题。
                3. 支持多轮对话。
                """
            )

if __name__ == "__main__":
    # 启动服务
    print("🚀 正在启动 Gradio 服务...")
    # Gradio 6.x: theme 参数移动到 launch 方法中
    # allowed_paths=["."] 允许访问当前目录下的文件（如图片）
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, theme=theme, allowed_paths=["."]) 
