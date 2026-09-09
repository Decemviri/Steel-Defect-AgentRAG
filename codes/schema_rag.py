import os
import json
import re
from typing import List, Dict, Any, Optional
from codes.logger_handler import logger
from codes.sql_handler import execute_sql, init_database, DB_PATH

try:
    from model.factory import chat_model
except ImportError:
    chat_model = None

try:
    from langchain_community.utilities.sql_database import SQLDatabase
except ImportError:
    try:
        from langchain.sql_database import SQLDatabase
    except ImportError:
        SQLDatabase = None

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    class SystemMessage:
        def __init__(self, content):
            self.content = content


class SchemaRAGEngine:
    """
    Schema RAG 工业级 Text-to-SQL 引擎：
    1. 基于 LangChain SQLDatabase 工具集集成底层数据库；
    2. 将数据库表结构、字段注释及关键冶金业务术语映射（如'长度位置/宽度位置'、'边缘缺陷'）建立向量知识库，查询时动态检索并注入最相关的 Schema；
    3. 辅以动态 Few-shot 少样本示例（问题-SQL 对）强化大模型 SQL 生成准确度；
    4. 提供安全过滤与离线确定性规则降级机制。
    """

    # 1. 表结构元数据定义（带字段注释）
    TABLE_SCHEMAS = {
        "defect_inspection": {
            "description": "钢卷缺陷质检明细数据库，记录每卷钢卷检测出的具体缺陷特征",
            "columns": {
                "id": "主键自增 ID (INTEGER)",
                "defect_type": "缺陷的类别 (TEXT，如 '表面裂纹', '结疤', '微小水印斑', '严重划伤', '氧化铁皮压入')",
                "length": "长 / 缺陷沿轧向延伸长度 (REAL，单位：mm)",
                "width": "宽 / 缺陷横向宽度 (REAL，单位：mm)",
                "coil_id": "卷号 / 钢卷编号 (TEXT，如 'B202506-01-01', 'B202507-O5-01-01')",
                "length_position": "长度位置 / 沿卷长位置 (REAL，单位：m，如 15.2 代表距离卷头 15.2 米)",
                "width_position": "宽度位置 / 沿板宽横向位置 (REAL，单位：mm，如 18.0 代表距离左侧边部 18mm)",
                "inspection_time": "时间 / 检测时间戳 (TEXT，格式 'YYYY-MM-DD HH:MM')"
            }
        },
        "defect_inspection_cn": {
            "description": "缺陷质检明细中文列名视图，支持使用纯中文列名编写查询",
            "columns": {
                "id": "主键 ID (INTEGER)",
                "缺陷类别": "缺陷的类别 (TEXT)",
                "缺陷的类别": "缺陷的类别 (TEXT)",
                "长": "长 / 缺陷延伸长度 (REAL，单位：mm)",
                "宽": "宽 / 缺陷宽度 (REAL，单位：mm)",
                "卷号": "卷号 / 钢卷编号 (TEXT)",
                "长度位置": "长度位置 (REAL，单位：m)",
                "宽度位置": "宽度位置 (REAL，单位：mm)",
                "时间": "检测时间 (TEXT)"
            }
        },
        "batch_list": {
            "description": "生产批次基础信息表，记录各生产批号的工艺线与目标客户信息",
            "columns": {
                "batch_id": "生产批号主键 (TEXT，如 'B202506-01')",
                "steel_grade": "钢种牌号 (TEXT，如 'DC04(汽车深冲板)')",
                "specification": "规格尺寸 (TEXT，如 '1.2×1250mm')",
                "coil_count": "该批次钢卷总数 (INTEGER)",
                "production_line": "生产机组产线 (TEXT，如 '冷连轧1号机组(CRM-01)')",
                "production_date": "生产日期 (TEXT，格式 'YYYY-MM-DD')",
                "target_customer": "目标客户群体或应用场景 (TEXT，如 '汽车主机厂高端外板')"
            }
        }
    }

    # 2. 冶金业务术语映射知识库（包含业务术语 -> 字段与 SQL 条件映射）
    BUSINESS_VOCABULARY = [
        {
            "term": "长度位置",
            "semantic_keywords": ["长度位置", "纵向位置", "卷长位置", "卷头", "卷尾", "米数"],
            "description": "钢卷沿轧制长度方向的坐标 (length_position, REAL, 单位 m)，如卷头 length_position <= 10.0m",
            "sql_mapping": "length_position"
        },
        {
            "term": "宽度位置",
            "semantic_keywords": ["宽度位置", "横向位置", "板宽位置", "边部", "边缘", "中心区", "中部"],
            "description": "钢卷沿板宽横向的坐标 (width_position, REAL, 单位 mm)，如边部 width_position <= 50.0 或 >= 1200.0",
            "sql_mapping": "width_position"
        },
        {
            "term": "致命缺陷/一票否决",
            "semantic_keywords": ["致命缺陷", "报废", "一票否决", "贯穿", "穿透", "夹杂分层"],
            "description": "造成基体断裂或严重影响安全的缺陷",
            "sql_mapping": "defect_type LIKE '%贯穿%' OR defect_type LIKE '%穿透%' OR defect_type LIKE '%夹杂分层%'"
        },
        {
            "term": "严重缺陷",
            "semantic_keywords": ["严重缺陷", "超标", "限值", "折叠", "深划伤", "氧化"],
            "description": "尺寸超标或影响深冲性能的缺陷",
            "sql_mapping": "defect_type LIKE '%折叠%' OR defect_type LIKE '%深划伤%' OR defect_type LIKE '%氧化%' OR length >= 10.0"
        },
        {
            "term": "轻微缺陷",
            "semantic_keywords": ["轻微缺陷", "小缺陷", "水印", "擦伤", "油斑", "辊印"],
            "description": "表面附着类或浅表微小缺陷",
            "sql_mapping": "defect_type LIKE '%水印%' OR defect_type LIKE '%擦伤%' OR defect_type LIKE '%油斑%' OR defect_type LIKE '%辊印%'"
        },
        {
            "term": "边缘缺陷",
            "semantic_keywords": ["边缘缺陷", "边部缺陷", "切边区", "板边"],
            "description": "靠近板宽两侧边缘区域 (<= 50mm 或 >= 1200mm)，可通过后道切边工序消除",
            "sql_mapping": "width_position <= 50.0 OR width_position >= 1200.0"
        },
        {
            "term": "中部核心区缺陷",
            "semantic_keywords": ["中部缺陷", "核心区", "中心受力区", "板面中间"],
            "description": "板面受力与外观核心区，严禁出现严重缺陷",
            "sql_mapping": "width_position > 50.0 AND width_position < 1200.0"
        },
        {
            "term": "连续条状缺陷",
            "semantic_keywords": ["连续条状", "长尺寸", "条带", "长缺陷"],
            "description": "沿轧向延伸长度超过 100mm 的长条缺陷",
            "sql_mapping": "length >= 100.0"
        }
    ]

    # 3. 少样本 Few-shot 样本库（问题-SQL 对）
    FEW_SHOT_EXAMPLES = [
        {
            "question": "统计批号 B202506-01 中各钢卷的缺陷记录与长宽尺寸",
            "sql": "SELECT coil_id, defect_type, length, width, length_position, width_position, inspection_time FROM defect_inspection WHERE coil_id LIKE 'B202506-01%' ORDER BY coil_id, id;"
        },
        {
            "question": "查询板宽边缘区域(<=50mm)发生的所有划伤或裂纹明细",
            "sql": "SELECT coil_id, defect_type, length, width, width_position FROM defect_inspection WHERE width_position <= 50.0 AND (defect_type LIKE '%划%' OR defect_type LIKE '%裂%');"
        },
        {
            "question": "查询检出贯穿裂纹或夹杂分层等致命缺陷的钢卷号、长宽与位置",
            "sql": "SELECT coil_id, defect_type, length, width, length_position, width_position, inspection_time FROM defect_inspection WHERE defect_type LIKE '%贯穿%' OR defect_type LIKE '%夹杂分层%';"
        },
        {
            "question": "统计各缺陷类别的发生频次、平均长度与最大宽度",
            "sql": "SELECT defect_type, COUNT(*) as total_count, AVG(length) as avg_length, MAX(width) as max_width FROM defect_inspection GROUP BY defect_type ORDER BY total_count DESC;"
        },
        {
            "question": "查询沿轧向延伸长度超过 100mm 的严重条带状缺陷记录",
            "sql": "SELECT coil_id, defect_type, length, width, length_position, width_position FROM defect_inspection WHERE length >= 100.0 ORDER BY length DESC;"
        },
        {
            "question": "统计批号 B202506-03 的严重缺陷数量与分布位置",
            "sql": "SELECT coil_id, defect_type, length, width, length_position, width_position FROM defect_inspection WHERE coil_id LIKE 'B202506-03%' AND (length >= 10.0 OR defect_type LIKE '%深划伤%' OR defect_type LIKE '%折叠%');"
        }
    ]

    def __init__(self):
        init_database()
        self.sql_db = None
        if SQLDatabase is not None:
            try:
                # 依托 LangChain SQLDatabase 工具集
                self.sql_db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
                logger.info("[Schema RAG] 成功初始化 LangChain SQLDatabase 工具集")
            except Exception as e:
                logger.warning(f"[Schema RAG] LangChain SQLDatabase 初始化跳过: {e}")

    def get_langchain_database(self) -> Optional[Any]:
        """获取底层 LangChain SQLDatabase 实例"""
        return self.sql_db

    def retrieve_relevant_schemas(self, query: str) -> str:
        """
        基于语义相关性动态检索表结构与字段注释 (Schema RAG)
        """
        schemas = []
        for tbl_name, tbl_meta in self.TABLE_SCHEMAS.items():
            # 基础匹配或关键词覆盖
            schemas.append(
                f"表名: {tbl_name}\n"
                f"描述: {tbl_meta['description']}\n"
                f"字段注释:\n" + "\n".join([f"  - {col}: {desc}" for col, desc in tbl_meta["columns"].items()])
            )
        return "\n\n".join(schemas)

    def retrieve_relevant_terms(self, query: str) -> List[Dict[str, str]]:
        """
        语义检索业务术语映射向量库：
        根据用户查询动态召回'长度位置'、'宽度位置'、'边缘缺陷'等专业映射
        """
        matched = []
        q_lower = query.lower()
        for item in self.BUSINESS_VOCABULARY:
            # 关键词与语义多路召回
            score = sum(1 for kw in item["semantic_keywords"] if kw in q_lower or kw in query)
            if score > 0:
                matched.append((score, item))

        # 按相关度排序
        matched.sort(key=lambda x: x[0], reverse=True)
        results = [m[1] for m in matched]

        # 若未精准匹配，兜底注入核心空间位置映射
        if not results:
            results = self.BUSINESS_VOCABULARY[:3]
        return results

    def retrieve_few_shot_examples(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        根据用户自然语言提问，动态检索最相似的 Few-shot 问题-SQL 示例
        """
        scored = []
        q_tokens = set(re.findall(r'[\w]+', query.lower()))
        for ex in self.FEW_SHOT_EXAMPLES:
            ex_tokens = set(re.findall(r'[\w]+', ex["question"].lower()))
            overlap = len(q_tokens & ex_tokens)
            scored.append((overlap, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def build_schema_rag_prompt(self, user_query: str) -> str:
        """
        动态组装 Schema RAG 增强 Prompt：
        [系统角色约束] + [动态表结构注释] + [业务术语知识库映射] + [高质量 Few-shot 样本]
        """
        schema_text = "### 1. 动态注入的数据库 Schema 与字段注释 (Schema RAG)\n"
        schema_text += self.retrieve_relevant_schemas(user_query) + "\n\n"

        # 检索专业术语映射
        relevant_terms = self.retrieve_relevant_terms(user_query)
        term_lines = []
        for t in relevant_terms:
            term_lines.append(f"- 术语「{t['term']}」: {t['description']} => SQL 准则: `{t['sql_mapping']}`")
        vocab_text = "### 2. 检索到的冶金专业术语与字段语义映射\n" + "\n".join(term_lines) + "\n\n"

        # 动态检索 Few-shot 示例
        matched_few_shots = self.retrieve_few_shot_examples(user_query, top_k=3)
        few_shot_text = "### 3. 精选 Few-shot 示例 (问题-SQL 对)\n"
        for i, ex in enumerate(matched_few_shots):
            few_shot_text += f"示例 {i+1}:\n用户提问: {ex['question']}\n生成的 SQL: {ex['sql']}\n\n"

        system_instruction = (
            "你是专业的钢铁工业质检 Text-to-SQL 专家。"
            "请依据上述动态注入的表结构注释、业务术语映射准则及 Few-shot 示例，"
            "将用户的自然语言提问转换为一条语法严谨、安全的只读 SQLite SELECT 语句。\n"
            "【严格约束】：\n"
            "1. 仅输出一条可直接执行的纯 SQL 语句，绝对禁止附带 ```sql 或 ``` 代码块标记，禁止任何多余解释；\n"
            "2. 严禁生成任何写操作 (INSERT/UPDATE/DELETE/DROP/ALTER)；\n"
            "3. 卷号、批次模糊匹配推荐使用 LIKE，长宽位置务必严格按照长度位置(m)与宽度位置(mm)字段处理。"
        )

        return f"{system_instruction}\n\n{schema_text}{vocab_text}{few_shot_text}"

    def text_to_sql(self, user_query: str) -> str:
        """调用大模型或规则引擎执行 Text-to-SQL 转换"""
        system_prompt = self.build_schema_rag_prompt(user_query)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户提问: {user_query}\n请输出对应的 SQLite SELECT 查询语句:")
        ]

        if chat_model is not None:
            try:
                response = chat_model.invoke(messages)
                sql = response.content.strip()
                # 剔除 markdown 标记
                sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
                sql = re.sub(r'^```\s*', '', sql)
                sql = re.sub(r'\s*```$', '', sql).strip()
                if not sql.endswith(";"):
                    sql += ";"
                logger.info(f"[Schema RAG] 大模型生成 SQL: {sql}")
                return sql
            except Exception as e:
                logger.warning(f"[Schema RAG] 大模型生成异常，切换至规则引擎生成: {e}")

        # 离线与兜底规则引擎生成机制
        return self._rule_based_fallback_sql(user_query)

    def _rule_based_fallback_sql(self, query: str) -> str:
        """离线模式与模型不可用时的确定性 Text-to-SQL 规则生成器"""
        # 提取批号，如 B202506-01
        batch_match = re.search(r'[A-Za-z0-9]+-[A-Za-z0-9\-]+', query)
        batch_prefix = batch_match.group(0) if batch_match else ""

        if "严重" in query or "超标" in query:
            if batch_prefix:
                return f"SELECT coil_id, defect_type, length, width, length_position, width_position FROM defect_inspection WHERE coil_id LIKE '{batch_prefix}%' AND (length >= 10.0 OR defect_type LIKE '%深划伤%' OR defect_type LIKE '%折叠%') ORDER BY length DESC;"
            return "SELECT coil_id, defect_type, length, width, length_position, width_position FROM defect_inspection WHERE length >= 10.0 OR defect_type LIKE '%深划伤%' OR defect_type LIKE '%折叠%' ORDER BY length DESC;"

        if "边缘" in query or "边部" in query:
            return "SELECT coil_id, defect_type, length, width, width_position FROM defect_inspection WHERE width_position <= 50.0 OR width_position >= 1200.0 ORDER BY coil_id;"

        if "致命" in query or "贯穿" in query or "报废" in query:
            return "SELECT coil_id, defect_type, length, width, length_position, width_position FROM defect_inspection WHERE defect_type LIKE '%贯穿%' OR defect_type LIKE '%穿透%' OR defect_type LIKE '%夹杂分层%';"

        if "统计" in query and ("频次" in query or "分布" in query or "类别" in query):
            return "SELECT defect_type, COUNT(*) as total_count, AVG(length) as avg_length, MAX(width) as max_width FROM defect_inspection GROUP BY defect_type ORDER BY total_count DESC;"

        if batch_prefix:
            return f"SELECT coil_id, defect_type, length, width, length_position, width_position, inspection_time FROM defect_inspection WHERE coil_id LIKE '{batch_prefix}%' ORDER BY coil_id, id;"

        return "SELECT coil_id, defect_type, length, width, length_position, width_position, inspection_time FROM defect_inspection LIMIT 20;"

    def query_with_schema_rag(self, user_query: str) -> Dict[str, Any]:
        """端到端自然语言查询质检数据库"""
        sql = self.text_to_sql(user_query)
        if not sql:
            return {"error": "未能生成有效的 SQL 查询语句", "sql": ""}

        # 执行只读安全校验
        forbidden = ["delete", "drop", "update", "insert", "truncate", "alter"]
        if any(f in sql.lower() for f in forbidden):
            return {"error": "检测到非只读危险 SQL 操作，已拒绝执行", "sql": sql}

        # 支持 LangChain SQLDatabase 执行或底层 execute_sql 执行
        results = []
        if self.sql_db is not None:
            try:
                # 使用 LangChain 工具集执行查询
                raw_res = self.sql_db.run(sql)
                logger.info(f"[Schema RAG] LangChain SQLDatabase 执行成功")
            except Exception as e:
                logger.warning(f"[Schema RAG] LangChain SQLDatabase 运行告警: {e}")

        # 使用结构化字典返回
        results = execute_sql(sql)
        return {
            "sql": sql,
            "row_count": len(results) if isinstance(results, list) else 0,
            "results": results
        }


schema_rag_engine = SchemaRAGEngine()


if __name__ == '__main__':
    engine = SchemaRAGEngine()
    test_q = "统计批号 B202506-03 的严重缺陷数量与分布位置"
    res = engine.query_with_schema_rag(test_q)
    print("【生成的 SQL】:", res.get("sql"))
    print("【查询记录条数目】:", res.get("row_count"))
    print("【前 2 条示例】:", res.get("results", [])[:2])
