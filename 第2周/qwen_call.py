import os
import sys
from http import HTTPStatus
from typing import Optional

from dashscope import Generation   #导入dashscope库,是python的第三方库,SDK,是用于调用阿里云的api的封装


def run_generation(prompt: str, api_key: str) -> Optional[str]:
    try:
        response = Generation.call(
            model="qwen-max",
            api_key=api_key,
            prompt=prompt,
        )
    except Exception as exc:  # 网络或 SDK 异常
        print(f"调用异常: {exc}", file=sys.stderr)
        return None

    if response.status_code == HTTPStatus.OK:
        return response.output.text

    print(
        f"调用失败，状态码: {response.status_code}, 错误信息: {response.message}",
        file=sys.stderr,
    )
    return None


def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("未找到环境变量 DASHSCOPE_API_KEY，请先配置 API Key。", file=sys.stderr)
        sys.exit(1)

    prompt = "你好，请介绍一下你自己"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    result = run_generation(prompt, api_key)
    if result is None:
        sys.exit(1)

    print("模型回复：")
    print(result)


if __name__ == "__main__":
    main()


