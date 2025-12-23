import os
import pathlib
import sys

# 开发时取项目根目录，打包后取 PyInstaller 临时目录
BASE_DIR = pathlib.Path(getattr(sys, '_MEIPASS', pathlib.Path(__file__).resolve().parent.parent))

def load_stylesheet(file: str) -> str:
    """读取 *.qss 并返回字符串"""
    qss_path = os.path.join(BASE_DIR, file)
    return pathlib.Path(qss_path).read_text(encoding="utf-8")