from http import HTTPStatus
from dashscope import MultiModalConversation
from PIL import Image
import os
# 本地图片路径
image_path = r'.\第29周\dog_playing_poker_20251216_160313.png'  # 请确保图片存在

resp = MultiModalConversation.call(
    model='qwen-vl-plus',
    messages=[{

        'role': 'user',
        'content': [
            {'image': f'file://{os.path.abspath(image_path)}'},
            {'text': '图片里有什么？请用中文简洁描述，并回答我提出的问题。'}
        ]
    }]
)

if resp.status_code == HTTPStatus.OK:
    # 打印完整的响应内容，或者提取 content
    try:
        content = resp.output.choices[0]['message']['content']
        # content 是一个列表，包含 text 和 box 等信息
        for item in content:
            if 'text' in item:
                print(item['text'])
    except Exception as e:
        print(f"解析响应出错: {e}")
        print(f"原始响应: {resp}")
else:
    print(resp.code, resp.message)