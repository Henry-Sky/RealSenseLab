from pathlib import Path

def load_stylesheet(file: str) -> str:
    """读取 *.qss 并返回字符串"""
    return Path(file).read_text(encoding="utf-8")