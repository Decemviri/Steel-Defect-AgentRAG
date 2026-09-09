import hashlib
import os
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader, TextLoader
    except ImportError:
        class TextLoader:
            def __init__(self, file_path, encoding="utf-8"):
                self.file_path = file_path
                self.encoding = encoding
            def load(self):
                with open(self.file_path, "r", encoding=self.encoding) as f:
                    content = f.read()
                from rag.hybrid_retriever import Document
                return [Document(page_content=content)]
        class PyPDFLoader:
            def __init__(self, file_path):
                self.file_path = file_path
            def load(self):
                return []

from .logger_handler import logger


def get_file_md5_hex(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return

    if not os.path.isfile(file_path):
        logger.error(f"不是文件: {file_path}")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5_obj.update(chunk)
                md5_hex = md5_obj.hexdigest()
                return md5_hex
    except Exception as e:
        logger.error(f"计算文件MD5失败: {file_path}")
from typing import Tuple, List, Any

def listdir_with_allowed_type(path: str, allowed_type: Tuple[str, ...]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"不是文件夹: {path}")
        return allowed_type
    for f in os.listdir(path):
        if f.endswith(allowed_type):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(filepath: str, passwd: str = None) -> List[Any]:
    return PyPDFLoader(filepath).load()


def txt_loader(filepath: str) -> List[Any]:
    return TextLoader(filepath).load()


