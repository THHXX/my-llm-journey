from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- 配置 ---
# 请将您的通义千问 API Key 填入此处，或者设置环境变量 DASHSCOPE_API_KEY
# 如果没有 Key，可以去阿里云申请
# os.environ["DASHSCOPE_API_KEY"] = "sk-xxxxxxxxxxxxxxxx"

def run_translation_demo():
    print("🚀 正在初始化 LangChain...")

    try:
        # 1. 初始化 LLM (大语言模型)
        # model="qwen-max" 是阿里云线上的强大模型
        llm = Tongyi(model="qwen-max") 
        
        # 2. 定义 Prompt 模板
        # 使用最新的 .from_template 方法
        prompt = PromptTemplate.from_template("将以下内容翻译成英文：{text}")
        
        # 3. 构建 Chain (链) - 使用最新的 LCEL (LangChain Expression Language) 语法
        # 这里的 | 符号就像管道一样，把数据从左传到右：
        # 输入 -> Prompt -> LLM -> 字符串输出解析器
        chain = prompt | llm | StrOutputParser()
        
        # 4. 验证任务：翻译 5 个不同句子
        sentences = [
            "你好，世界",
            "LangChain 是一个开发大语言模型应用的框架。",
            "今天天气真不错，适合出去散步。",
            "人工智能正在改变我们的生活。",
            "学习编程需要坚持不懈的努力。"
        ]
        
        print("\n📝 开始翻译任务：\n")
        
        for i, s in enumerate(sentences, 1):
            print(f"[{i}/5] 原文：{s}")
            # 运行链，使用 .invoke() 方法
            result = chain.invoke({"text": s})
            print(f"      译文：{result.strip()}\n")
            
        print("✅ 所有任务完成！")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("💡 提示：请检查是否设置了 DASHSCOPE_API_KEY，以及是否安装了 dashscope 库。")

if __name__ == "__main__":
    run_translation_demo()
