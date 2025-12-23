from pathlib import Path

def lab_read_path(file_path : str) -> str:
    """
    检查路径中文件是否存在，方便分发时打包资源文件
    :param file_path:
    :return: path (路径存在返回路径str)
    """
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