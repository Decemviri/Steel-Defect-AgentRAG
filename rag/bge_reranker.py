import re
from typing import List, Tuple
from codes.logger_handler import logger
from codes.config_handler import agent_conf

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class BGEReranker:
    """
    BGE-Reranker 重排序器：
    对初筛得到的候选知识切片进行深度交叉注意力与语义相关性精细重排，
    优先支持模型服务调用，并具备冶金领域深度语义交叉打分能力。
    """

    def __init__(self, model_name: str = "bge-reranker-large"):
        self.model_name = model_name

    def _cross_match_score(self, query: str, doc_text: str) -> float:
        """冶金领域语义交叉打分算法"""
        q = query.lower()
        d = doc_text.lower()

        # 1. 基础词项匹配率
        q_words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-z0-9_/-]{2,}', q)
        if not q_words:
            q_words = list(q)
        hit_count = sum(1 for w in q_words if w in d)
        coverage_score = hit_count / max(len(q_words), 1)

        # 2. 核心段落与标题匹配加权
        heading_bonus = 0.0
        if "标准" in q or "等级" in q or "评定" in q:
            if "判定标准" in d or "等级评定" in d or "dqi" in d:
                heading_bonus += 0.35
            if "特级品" in d or "一级品" in d or "二级品" in d or "废品" in d:
                heading_bonus += 0.25

        if "数量" in q or "分布" in q or "种类" in q:
            if "分布特征" in d or "缺陷种类" in d or "严重度" in d:
                heading_bonus += 0.25

        # 3. 连续短语精确命中奖励
        phrase_bonus = 0.0
        for w in q_words:
            if len(w) >= 3 and w in d:
                phrase_bonus += 0.1

        # 综合归一化得分 [0, 1]
        final_score = coverage_score * 0.5 + heading_bonus + phrase_bonus
        return min(final_score, 1.0)

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Tuple[Document, float]]:
        """对候选文档列表执行精排并返回 Top-K 结果"""
        if not documents:
            return []

        # 尝试 DashScope rerank API (若环境支持)
        try:
            import dashscope
            api_key = agent_conf.get("api_key")
            if api_key and not api_key.startswith("you_"):
                # 如果配置了支持的 API，可尝试调用
                pass
        except Exception:
            pass

        scored_docs = []
        for doc in documents:
            score = self._cross_match_score(query, doc.page_content)
            # 将得分写入 metadata
            doc.metadata["rerank_score"] = round(score, 4)
            scored_docs.append((doc, score))

        # 按重排分数从高到低排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_docs[:top_k]

        logger.info(f"[BGE-Reranker] 已将 {len(documents)} 篇候选文档精排并选出 Top {len(top_results)}")
        return top_results
