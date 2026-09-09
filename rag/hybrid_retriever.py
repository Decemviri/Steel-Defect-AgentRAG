from typing import List
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}
from codes.logger_handler import logger
from codes.path_tool import get_abs_path
from rag.bm25_retriever import BM25Retriever
from rag.bge_reranker import BGEReranker
try:
    from rag.vector_store import VectorStoreService
except ImportError:
    class VectorStoreService:
        def get_retriever(self):
            class DummyRetriever:
                def invoke(self, q):
                    return []
            return DummyRetriever()


class HybridRerankRetriever:
    """
    基于“向量 + BM25”混合检索与 BGE-Reranker 重排序的高阶检索流水线
    """

    def __init__(self, dense_k: int = 6, sparse_k: int = 6, final_k: int = 3, rrf_k: int = 60):
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.final_k = final_k
        self.rrf_k = rrf_k

        self.vector_store_service = VectorStoreService()
        self.vector_retriever = self.vector_store_service.get_retriever()

        self.bm25_retriever = BM25Retriever()
        self.reranker = BGEReranker()
        self._is_bm25_fitted = False

        self._init_and_fit_bm25()

    def _init_and_fit_bm25(self):
        """装载并建立 BM25 索引"""
        try:
            # 优先从标准规范文本构建切片
            standards_path = get_abs_path("data/steel_coil_grading_standards.txt")
            import os
            if os.path.exists(standards_path):
                with open(standards_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # 按照章节与双换行切分成知识块
                sections = content.split("\n## ")
                docs = []
                for i, sec in enumerate(sections):
                    clean_sec = sec.strip()
                    if clean_sec:
                        prefix = "## " if i > 0 else ""
                        docs.append(Document(
                            page_content=prefix + clean_sec,
                            metadata={"source": "steel_coil_grading_standards.txt", "section": i}
                        ))
                self.bm25_retriever.fit(docs)
                self._is_bm25_fitted = True
        except Exception as e:
            logger.warning(f"[HybridRetriever] 初始化 BM25 失败: {str(e)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        执行完整检索流水线：
        1. 向量稠密检索召回
        2. BM25 稀疏检索召回
        3. RRF (Reciprocal Rank Fusion) 排名融合
        4. BGE-Reranker 交叉相关性精排
        """
        k = top_k or self.final_k

        # 1. 向量稠密检索
        dense_docs = []
        try:
            dense_docs = self.vector_retriever.invoke(query)[:self.dense_k]
        except Exception as e:
            logger.warning(f"[HybridRetriever] 向量检索调用异常: {str(e)}")

        # 2. BM25 稀疏检索
        sparse_docs = []
        if self._is_bm25_fitted:
            sparse_hits = self.bm25_retriever.search(query, top_k=self.sparse_k)
            sparse_docs = [doc for doc, _ in sparse_hits]

        # 3. Reciprocal Rank Fusion (RRF)
        doc_map = {}
        rrf_scores = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = doc.page_content[:60]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (0.5 / (self.rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.page_content[:60]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (0.5 / (self.rrf_k + rank + 1))

        # 排序候选集
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        candidate_docs = [doc_map[cid] for cid in sorted_candidates[:max(self.dense_k, self.sparse_k)]]

        # 若候选集为空，执行兜底
        if not candidate_docs and self.bm25_retriever.documents:
            candidate_docs = self.bm25_retriever.documents[:self.final_k]

        # 4. BGE-Reranker 精排
        reranked_tuples = self.reranker.rerank(query, candidate_docs, top_k=k)
        final_docs = [doc for doc, _ in reranked_tuples]

        logger.info(f"[HybridRetriever] 混合检索完成：稠密召回 {len(dense_docs)} 篇，BM25 召回 {len(sparse_docs)} 篇，精排输出 {len(final_docs)} 篇")
        return final_docs


if __name__ == '__main__':
    retriever = HybridRerankRetriever()
    results = retriever.retrieve("冷轧板深划伤缺陷判定标准与等级要求")
    for i, r in enumerate(results):
        print(f"--- Top {i+1} (Rerank score: {r.metadata.get('rerank_score')}) ---")
        print(r.page_content[:150])
