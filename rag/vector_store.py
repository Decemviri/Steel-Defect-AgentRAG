import os.path

from codes.config_handler import chroma_conf
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        try:
            from langchain.vectorstores import Chroma
        except ImportError:
            class Chroma:
                def __init__(self, *args, **kwargs):
                    pass
                def as_retriever(self, *args, **kwargs):
                    class DummyRetriever:
                        def invoke(self, q):
                            return []
                    return DummyRetriever()

from codes.logger_handler import logger
from codes.path_tool import get_abs_path
from model.factory import embeddings_model

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:
            def __init__(self, *args, **kwargs):
                pass
            def split_documents(self, docs):
                return docs
from codes.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embeddings_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chroma_conf["chunk_size"],
                                                       chunk_overlap=chroma_conf["chunk_overlap"],
                                                       separators=chroma_conf["separator"],
                                                       length_function=len)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_documents(self):
        def check_file_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                md5_hex_list = f.readlines()
                for md5_hex in md5_hex_list:
                    if md5_hex.strip() == md5_for_check:
                        return True

                return False

        def save_file_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check)
                f.write("\n")

        def get_file_documents(file_path: str):
            if file_path.endswith(".pdf"):
                return pdf_loader(file_path)
            elif file_path.endswith(".txt"):
                return txt_loader(file_path)
            else:
                logger.error(f"不支持的文件类型: {file_path}")
                return []

        # allowed_type = listdir_with_allowed_type(chroma_conf["data_path"],
        #                                          tuple(chroma_conf["allow_knowledge_file_type"]))

        allowed_type = listdir_with_allowed_type(get_abs_path(chroma_conf["data_path"]),
                                                 tuple(chroma_conf["allow_knowledge_file_type"]))
        for path in allowed_type:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                continue
            if check_file_hex(md5_hex):
                logger.info(f"文件已存在: {path}")
                continue
            documents = get_file_documents(path)
            if not documents:
                logger.error(f"文件加载失败: {path}")
                continue
            texts = self.splitter.split_documents(documents)
            self.vector_store.add_documents(texts)
            save_file_hex(md5_hex)


if __name__ == '__main__':
    VectorStoreService().load_documents()
    retriver = VectorStoreService().get_retriever()
    res=retriver.invoke("迷路")
    for r in res:
        print("----"*20)
        print(r)
