import math
import re
from typing import List, Tuple, Any
from codes.logger_handler import logger

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class BM25Retriever:
    """适用于钢铁冶金专业知识切片的 BM25 稀疏检索器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_len: List[int] = []
        self.doc_freqs: List[dict] = []
        self.idf: dict = {}

    def _tokenize(self, text: str) -> List[str]:
        """专业中英文混合分词：按中文字符、连续英文/数字、冶金专有名词分词"""
        text = text.lower()
        # 提取中文字符、连续英文字母/数字
        tokens = re.findall(r'[\u4e00-\u9fa5]|[a-z0-9_/-]+', text)
        # 补充双字中文字元 (2-gram) 提升冶金专有名词召回率 (如 "裂纹", "结疤", "折叠", "深冲")
        zh_chars = [t for t in tokens if re.match(r'[\u4e00-\u9fa5]', t)]
        ngrams = [zh_chars[i] + zh_chars[i+1] for i in range(len(zh_chars) - 1)]
        return tokens + ngrams

    def fit(self, documents: List[Document]):
        """根据文档列表构建 BM25 倒排索引与 IDF 参数"""
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_len = []
        self.doc_freqs = []
        df = {}

        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length

            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)

            for token in freqs.keys():
                df[token] = df.get(token, 0) + 1

        self.avgdl = total_len / self.corpus_size

        self.idf = {}
        for token, freq in df.items():
            # BM25 IDF 平滑公式
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        logger.info(f"[BM25] 成功为 {self.corpus_size} 个文档块建立词典索引，词项总数: {len(self.idf)}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """检索与查询词匹配度最高的文档及其 BM25 得分"""
        if self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_l = self.doc_len[i]
            for token in query_tokens:
                if token not in freqs:
                    continue
                tf = freqs[token]
                idf = self.idf.get(token, 0.0)
                # BM25 核心权重计算
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avgdl))
                score += idf * (tf * (self.k1 + 1.0) / denom)
            if score > 0:
                scores.append((self.documents[i], score))

        # 按得分从高到低排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
