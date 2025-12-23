import sys
from pathlib import Path

# 开发时取项目根目录，打包后取 PyInstaller 临时目录
BASE_DIR = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))

def lab_read_path(file_path : str) -> Path:
    """
    检查路径中文件是否存在，方便分发时打包资源文件
    :param file_path:
    :return: path (路径存在返回路径str)
    """
    file_path = BASE_DIR / file_path
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return file_path

def load_file_content(file_path : str, encoding="utf-8") -> str:
    """
    读取文件内容
    :param file_path: 文件路径
    :param encoding: 文件编码格式
    :return: str (文件内容)
    """
    file_path = lab_read_path(file_path)
    return Path(file_path).read_text(encoding=encoding)