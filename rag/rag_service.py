import os
from typing import List, Any
from codes.path_tool import get_abs_path
from codes.logger_handler import logger

try:
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
except ImportError:
    try:
        from langchain.docstore.document import Document
        from langchain.schema.output_parser import StrOutputParser
        from langchain.prompts import PromptTemplate
    except ImportError:
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}
        class StrOutputParser:
            def parse(self, x):
                return getattr(x, "content", str(x))
        class PromptTemplate:
            def __init__(self, template: str = ""):
                self.template = template
            @classmethod
            def from_template(cls, t):
                return cls(t)
            def format(self, **kwargs):
                return self.template.format(**kwargs)

from codes.prompt_loader import load_rag_prompts
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRerankRetriever


def print_prompt(prompt):
    logger.debug("Prompt generated for RAG summary")
    return prompt


class RagSummaryService:
    def __init__(self):
        self.hybrid_retriever = HybridRerankRetriever()
        self.prompt_text = load_rag_prompts()
        try:
            self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        except Exception:
            self.prompt_template = None
        self.model = chat_model

    def retriver_docs(self, query: str) -> List[Any]:
        try:
            # 采用“向量 + BM25”混合检索与 BGE-Reranker 精排
            return self.hybrid_retriever.retrieve(query, top_k=3)
        except Exception as e:
            logger.warning(f"混合检索与重排调用异常: {str(e)}")
            return []

    def generate_summary(self, query: str) -> str:
        docs = self.retriver_docs(query)
        context = ""
        count = 0
        for doc in docs:
            count += 1
            content = getattr(doc, "page_content", str(doc))
            meta = getattr(doc, "metadata", {})
            context += f"【参考资料{count}】:{content}|参考元数据:{meta}\n"

        # 若向量库中暂无检索结果，自动从本地规则标准文件回退补充上下文
        if not context.strip():
            standards_path = get_abs_path("data/steel_coil_grading_standards.txt")
            if os.path.exists(standards_path):
                try:
                    with open(standards_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        context = f"【钢卷缺陷等级判定标准规范】:\n{content[:4000]}"
                except Exception as e:
                    logger.error(f"读取本地规则库失败: {str(e)}")

        if self.model is not None and self.prompt_template is not None:
            try:
                chain = self.prompt_template | self.model | StrOutputParser()
                return chain.invoke({
                    "input": query,
                    "context": context,
                })
            except Exception as e:
                logger.warning(f"大模型总结生成异常，返回原始知识上下文: {e}")

        return f"【知识库混合检索结果 ({query})】:\n{context}"


if __name__ == '__main__':
    rag_service = RagSummaryService()
    print(rag_service.generate_summary("热轧带钢表面结疤缺陷的产生原因及预防措施"))
