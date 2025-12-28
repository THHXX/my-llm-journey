"""演示文件读写。"""

from pathlib import Path


def write_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f]


if __name__ == "__main__":
    demo_path = Path("basics_demo.txt")
    write_text(demo_path, "第一行\n第二行\n第三行")
    print("全文读取:")
    print(read_text(demo_path))
    print("逐行读取:")
    print(read_lines(demo_path))

