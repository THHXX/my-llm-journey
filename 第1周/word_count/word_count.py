"""简单词频统计脚本。

功能：
- 读取文本文件
- 规范化文本（小写、去标点）
- 统计词频并输出 Top N
- 将完整结果写入输出文件
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PUNCTUATION = ",.!?;:\"'()[]{}<>/\\|@#$%^&*-_=+`~"


def read_text(path: Path) -> str:
    """读取文本内容。"""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """小写并去掉常见标点，保留空白用于分词。"""
    lowered = text.lower()
    cleaned = lowered
    for ch in PUNCTUATION:
        cleaned = cleaned.replace(ch, " ")
    return cleaned


def tokenize(text: str) -> List[str]:
    """按空白分词，过滤空词。"""
    return [word for word in text.split() if word]


def count_words(words: Iterable[str]) -> Dict[str, int]:
    """统计词频。"""
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def top_n(freq: Dict[str, int], n: int = 10) -> List[Tuple[str, int]]:
    """返回按频次排序的前 N 个词。"""
    return sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:n]


def save_result(freq: Dict[str, int], output_path: Path) -> None:
    """将完整词频结果写入文件（按频次降序、词典序升序）。"""
    sorted_items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    with output_path.open("w", encoding="utf-8") as f:
        for word, count in sorted_items:
            f.write(f"{word} {count}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Word frequency counter")
    parser.add_argument(
        "--input",
        default="./word_count/input.txt",
        help="输入文本路径（默认 word_count/input.txt）",
    )
    parser.add_argument(
        "--output",
        default="./word_count/output.txt",
        help="输出结果路径（默认 word_count/output.txt）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="终端展示的 Top N 词频，默认 10",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    text = read_text(input_path)
    cleaned = clean_text(text)
    words = tokenize(cleaned)
    freq = count_words(words)

    # 终端输出 Top N
    print(f"Total words: {len(words)} | Unique words: {len(freq)}")
    print(f"Top {args.top}:")
    for word, count in top_n(freq, args.top):
        print(f"{word}: {count}")

    # 写入完整结果
    save_result(freq, output_path)
    print(f"All results written to: {output_path}")


if __name__ == "__main__":
    main()

