import os
from http import HTTPStatus
from dashscope import MultiModalConversation

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

def ask_image(image_path: str, prompt: str) -> str:
    if not os.path.exists(image_path):
        return "错误：找不到图片文件"
    resp = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=[{
            'role': 'user',
            'content': [
                {'image': f'file://{os.path.abspath(image_path)}'},
                {'text': prompt}
            ]
        }]
    )
    if hasattr(resp, 'status_code') and resp.status_code == HTTPStatus.OK:
        try:
            content = resp.output.choices[0]['message']['content']
            for item in content:
                if 'text' in item:
                    return item['text']
            return str(content)
        except Exception:
            return str(resp)
    return f"{getattr(resp,'code','')} {getattr(resp,'message','')}"

if __name__ == "__main__":
    img = r".\第30周\example.png"
    q = "请用中文详细描述图片，并回答我提出的复杂问题"
    print(ask_image(img, q))
