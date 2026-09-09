import os
import json
import re
from typing import Optional, List, Dict, Any

# 初始化 LangSmith 全链路可观测性与追踪配置
from codes.config_handler import agent_conf
ls_conf = agent_conf.get("langsmith") if isinstance(agent_conf.get("langsmith"), dict) else {}
tracing_enabled = False
if isinstance(ls_conf, dict) and ls_conf.get("tracing"):
    tracing_enabled = True
    project_name = ls_conf.get("project", "Steel-Defect-AgentRAG")
    api_key = ls_conf.get("api_key")
else:
    tracing_enabled = (agent_conf.get("tracing") in [True, "true"]) or (agent_conf.get("langsmith_tracing") in [True, "true"])
    project_name = agent_conf.get("project") or agent_conf.get("langsmith_project") or "Steel-Defect-AgentRAG"
    api_key = agent_conf.get("api_key") or agent_conf.get("langsmith_api_key")

if tracing_enabled:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = str(project_name)
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = str(api_key)

try:
    from typing import TypedDict
except ImportError:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        TypedDict = dict

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"
    class StateGraph:
        def __init__(self, state_schema):
            self.nodes = {}
            self.edges = {}
            self.cond_edges = {}
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, from_node, to_node):
            self.edges[from_node] = to_node
        def add_conditional_edges(self, from_node, condition, mapping):
            self.cond_edges[from_node] = (condition, mapping)
        def compile(self):
            class CompiledGraph:
                def __init__(self, nodes, edges, cond_edges):
                    self.nodes = nodes
                    self.edges = edges
                    self.cond_edges = cond_edges
                def invoke(self, state):
                    curr = self.edges.get(START, "sql_query")
                    while curr != END and curr is not None:
                        func = self.nodes.get(curr)
                        if func:
                            updates = func(state)
                            if updates:
                                state.update(updates)
                        if curr in self.cond_edges:
                            cond_func, mapping = self.cond_edges[curr]
                            dest_key = cond_func(state)
                            curr = mapping.get(dest_key, END)
                        elif curr in self.edges:
                            curr = self.edges[curr]
                        else:
                            break
                    return state
            return CompiledGraph(self.nodes, self.edges, self.cond_edges)

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    class SystemMessage:
        def __init__(self, content):
            self.content = content
    class HumanMessage:
        def __init__(self, content):
            self.content = content

try:
    from pydantic import ValidationError
except ImportError:
    class ValidationError(Exception):
        pass

from codes.logger_handler import logger
from codes.sql_handler import query_batch_defects_analysis, format_batch_defects_report_context
try:
    from model.factory import chat_model
except ImportError:
    chat_model = None

from model.schema import (
    SteelInspectionReport,
    GradingEvaluation,
    DefectItem,
    MetallurgicalAnalysis,
    ActionPlan
)
from rag.hybrid_retriever import HybridRerankRetriever


class ReportWorkflowState(TypedDict):
    """LangGraph 状态图全生命周期状态定义"""
    batch_id: str
    user_query: str
    sql_data: Dict[str, Any]
    sql_text: str
    rag_standards: str
    raw_llm_output: str
    parsed_report: Optional[Dict[str, Any]]
    final_markdown: str
    consistency_passed: bool
    consistency_error: Optional[str]
    retry_count: int
    max_retries: int
    error_message: Optional[str]
    is_degraded: bool


class SteelInspectionGraphPipeline:
    """
    工业级基于 LangGraph 状态机的钢卷质检评级与报告生成流水线：
    1. SQL 结构化质检数据提取
    2. “向量 + BM25” 双路混合检索与 BGE-Reranker 重排标准知识注入
    3. Pydantic with_structured_output 强约束输出，杜绝格式幻觉
    4. 一致性交叉核验节点 (Consistency Check Node)：严密比对报表数值与 SQL 查询，杜绝数值幻觉
    5. LangGraph 条件边实现自动纠错重试与确定性分支降级兜底
    6. LangSmith 全链路追踪监控
    """

    def __init__(self):
        self.retriever = HybridRerankRetriever()
        self.graph = self._build_graph()

    def _node_sql_query(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 1：调用 SQL 引擎获取批号缺陷全貌"""
        batch_id = state.get("batch_id", "B202506-01")
        logger.info(f"[LangGraph: SQL Node] 执行批号 {batch_id} 的 SQL 质检数据聚合...")
        analysis = query_batch_defects_analysis(batch_id)
        sql_text = format_batch_defects_report_context(analysis)
        return {
            "sql_data": analysis,
            "sql_text": sql_text
        }

    def _node_hybrid_rag(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 2：执行“向量+BM25+BGE重排”检索对应的钢卷等级标准"""
        steel_grade = state.get("sql_data", {}).get("batch_meta", {}).get("steel_grade", "钢卷")
        query = f"{steel_grade} 钢卷表面缺陷 数量 种类 分布 等级判断标准 DQI 评级规则"
        logger.info(f"[LangGraph: Hybrid RAG Node] 检索标准: {query}")
        docs = self.retriever.retrieve(query, top_k=3)
        standards_text = "\n\n".join([f"【评级标准参考】:\n{d.page_content}" for d in docs])
        return {
            "rag_standards": standards_text
        }

    def _node_generate_report(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 3：调用大模型生成符合 Pydantic 结构的 JSON 报告 (优先 with_structured_output)"""
        pydantic_schema_json = json.dumps(SteelInspectionReport.model_json_schema(), ensure_ascii=False, indent=2)

        system_prompt = (
            "你是专业的钢铁质量检验与等级评定专家系统。\n"
            "根据提供的 SQL 质检数据与 RAG 钢卷等级判定标准，生成一份结构极其严谨的质检与评级报告。\n"
            "【严格约束】：\n"
            "1. 必须输出且仅输出一个合法的 JSON 对象，符合提供的 JSON Schema 规范；\n"
            "2. 严禁使用任何 markdown 代码块标记 (如 ```json 或 ```)，直接以 { 开头，以 } 结尾；\n"
            "3. 报告中的 grading.final_grade 必须严格从以下 5 个枚举中选取一个：['特级品', '一级品', '二级品', '协议品', '废品']；\n"
            "4. 报告正文 report_markdown 需是一篇排版精美的 Markdown 全文，涵盖六大规范板块（基本信息、SQL明细、标准多维对照、定级与DQI、冶金机理、处置对策）；\n"
            "5. 【严禁数值幻觉】：报告中的缺陷总数、各严重级别数量必须与提供的 SQL 真实数据 100% 绝对一致！"
        )

        user_content = (
            f"=== 1. 批次质检 SQL 真实数据 ===\n{state.get('sql_text')}\n\n"
            f"=== 2. 钢卷等级判断标准知识库 ===\n{state.get('rag_standards')}\n\n"
            f"=== 3. 必须遵循的 JSON Schema 结构 ===\n{pydantic_schema_json}\n\n"
            "请输出合法的 JSON 报告对象:"
        )

        try:
            if chat_model is None:
                raise RuntimeError("chat_model 未初始化或离线模式")

            # 优先使用 LangChain 官方 with_structured_output 模式强约束输出
            if hasattr(chat_model, "with_structured_output"):
                try:
                    structured_llm = chat_model.with_structured_output(SteelInspectionReport)
                    res_obj = structured_llm.invoke(user_content)
                    if isinstance(res_obj, SteelInspectionReport):
                        return {
                            "raw_llm_output": json.dumps(res_obj.model_dump(), ensure_ascii=False),
                            "parsed_report": res_obj.model_dump(),
                            "final_markdown": res_obj.report_markdown
                        }
                    elif isinstance(res_obj, dict):
                        return {"raw_llm_output": json.dumps(res_obj, ensure_ascii=False)}
                except Exception as ex:
                    logger.info(f"[with_structured_output] 降级至标准 JSON 提示词模式: {ex}")

            response = chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ])
            raw_output = response.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output, flags=re.IGNORECASE)
            raw_output = re.sub(r'^```\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output).strip()
            return {"raw_llm_output": raw_output}
        except Exception as e:
            logger.warning(f"[LangGraph: Generate Node] 模型生成阶段异常: {str(e)}，将转入降级分支")
            return {
                "raw_llm_output": "",
                "error_message": str(e),
                "retry_count": state.get("max_retries", 2) + 1
            }

    def _node_validate_pydantic(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 4：利用 Pydantic 强校验 JSON 输出结构与枚举类型"""
        if state.get("parsed_report") is not None:
            return {
                "parsed_report": state["parsed_report"],
                "final_markdown": state.get("final_markdown") or state["parsed_report"].get("report_markdown", ""),
                "error_message": None
            }

        raw_json = state.get("raw_llm_output", "")
        logger.info(f"[LangGraph: Validate Node] 正在进行 Pydantic 强约束校验 (重试次数: {state.get('retry_count')})...")

        try:
            if not raw_json.strip():
                raise ValueError("模型输出为空")
            data = json.loads(raw_json)
            report_obj = SteelInspectionReport.model_validate(data)
            logger.info(f"[LangGraph: Validate Node] Pydantic 校验通过！判定等级: {report_obj.grading.final_grade}")
            return {
                "parsed_report": report_obj.model_dump(),
                "final_markdown": report_obj.report_markdown,
                "error_message": None
            }
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            err_msg = str(e)
            current_retries = state.get("retry_count", 0) + 1
            logger.warning(f"[LangGraph: Validate Node] Pydantic 校验失败: {err_msg[:100]}... 累积重试: {current_retries}")
            return {
                "error_message": err_msg,
                "retry_count": current_retries
            }

    def _node_consistency_check(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 5：一致性交叉核验节点（核对报表中所有数值与 SQL 查询真实结果，杜绝数值幻觉）"""
        parsed_report = state.get("parsed_report")
        if not parsed_report:
            return {
                "consistency_passed": False,
                "consistency_error": "尚未通过 Pydantic 结构化解析"
            }

        sql_data = state.get("sql_data", {})
        sql_total = sql_data.get("total_defects", 0)

        # 提取 SQL 真实各级别统计
        sev_counts = {}
        for s in sql_data.get("severity_stats", []):
            sev_counts[s.get("defect_severity", "")] = s.get("total_count", 0)
        sql_fatal = sev_counts.get("A类致命缺陷", 0)
        sql_severe = sev_counts.get("B类严重缺陷", 0)
        sql_minor = sev_counts.get("C类轻微缺陷", 0)

        # 提取报表中的统计数据
        rep_total = parsed_report.get("total_defects", -1)
        grading = parsed_report.get("grading", {})
        rep_fatal = grading.get("fatal_defects_count", -1)
        rep_severe = grading.get("severe_defects_count", -1)
        rep_minor = grading.get("minor_defects_count", -1)

        discrepancies = []
        if rep_total != sql_total:
            discrepancies.append(f"缺陷总数不一致 (报告={rep_total}, 真实SQL={sql_total})")
        if rep_fatal != sql_fatal:
            discrepancies.append(f"A类致命缺陷数不一致 (报告={rep_fatal}, 真实SQL={sql_fatal})")
        if rep_severe != sql_severe:
            discrepancies.append(f"B类严重缺陷数不一致 (报告={rep_severe}, 真实SQL={sql_severe})")
        if rep_minor != sql_minor:
            discrepancies.append(f"C类轻微缺陷数不一致 (报告={rep_minor}, 真实SQL={sql_minor})")

        if discrepancies:
            err_feedback = "【数据一致性交叉核查失败(检测到数值幻觉)】：" + "；".join(discrepancies) + "。必须严格基于 SQL 真实数据，严禁产生数值幻觉！"
            logger.warning(f"[LangGraph: Consistency Check Node] {err_feedback}")
            current_retries = state.get("retry_count", 0) + 1
            return {
                "consistency_passed": False,
                "consistency_error": err_feedback,
                "error_message": err_feedback,
                "parsed_report": None,  # 判定报告无效，强制触发反哺重试
                "retry_count": current_retries
            }

        logger.info("[LangGraph: Consistency Check Node] ✅ 一致性核验通过：报表所有数值与 SQL 查询结果 100% 严格吻合！")
        return {
            "consistency_passed": True,
            "consistency_error": None
        }

    def _node_retry_with_feedback(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 6：校验或一致性核对失败后，携带具体错误反哺提示进行修正重试"""
        logger.info(f"[LangGraph: Retry Node] 正在执行第 {state.get('retry_count')} 次修正重试...")
        err_msg = state.get("error_message") or state.get("consistency_error") or "格式校验不合法"
        pydantic_schema_json = json.dumps(SteelInspectionReport.model_json_schema(), ensure_ascii=False, indent=2)

        prompt = (
            f"你上一次输出的内容未通过严格校验或一致性检查，具体原因如下：\n"
            f"【错误反馈提示】：{err_msg}\n\n"
            f"请仔细对照以下 JSON Schema 及 SQL 真实数据，修正字段名称、数据类型或统计数值，重新输出纯 JSON 对象：\n"
            f"{pydantic_schema_json}\n\n"
            f"SQL 真实数据参照：\n{state.get('sql_text')[:600]}"
        )

        try:
            if chat_model is None:
                raise RuntimeError("chat_model 未初始化或离线模式")
            response = chat_model.invoke([HumanMessage(content=prompt)])
            raw_output = response.content.strip()
            raw_output = re.sub(r'^```json\s*', '', raw_output, flags=re.IGNORECASE)
            raw_output = re.sub(r'^```\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output).strip()
            return {"raw_llm_output": raw_output}
        except Exception as e:
            logger.warning(f"[LangGraph: Retry Node] 重试调用异常: {str(e)}")
            return {
                "raw_llm_output": "",
                "error_message": str(e),
                "retry_count": state.get("max_retries", 2) + 1
            }

    def _node_fallback_degradation(self, state: ReportWorkflowState) -> Dict[str, Any]:
        """节点 7：重试超限自适应分支降级（规则引擎确定性兜底，保证100%可靠性）"""
        logger.warning("[LangGraph: Fallback Node] 重试次数超限，触发自适应分支降级（确定性规则引擎兜底）...")
        sql_data = state.get("sql_data", {})
        batch_id = state.get("batch_id", "未知批号")
        batch_meta = sql_data.get("batch_meta", {})
        steel_grade = batch_meta.get("steel_grade", "未知牌号")
        total_defects = sql_data.get("total_defects", 0)

        # 规则引擎计算 DQI 与定级
        fatal_count = 0
        severe_count = 0
        minor_count = 0

        for s in sql_data.get("severity_stats", []):
            sev = s.get("defect_severity", "")
            cnt = s.get("total_count", 0)
            if "A类" in sev or "致命" in sev:
                fatal_count += cnt
            elif "B类" in sev or "严重" in sev:
                severe_count += cnt
            else:
                minor_count += cnt

        # 估算 DQI
        dqi = fatal_count * 100.0 + severe_count * 15.0 + minor_count * 2.0

        # 定级裁决
        if fatal_count > 0 or dqi > 150.0:
            final_grade = "废品"
            decision = "判废拒绝出库，送返熔处理或回炉退火"
        elif severe_count > 6 or dqi > 80.0:
            final_grade = "协议品"
            decision = "经技术评审降为协议品，建议用户切头切边后折价使用"
        elif severe_count > 2 or dqi > 30.0:
            final_grade = "二级品"
            decision = "降级为二级品使用，用于非外露结构冲压件"
        elif severe_count > 0 or minor_count > 1 or dqi > 5.0:
            final_grade = "一级品"
            decision = "符合商用合格品标准，准予正常出库发货"
        else:
            final_grade = "特级品"
            decision = "达到汽车外板 O5 优级品标准，直接发往汽车主机厂"

        degraded_markdown = f"""# 钢卷缺陷质检与质量等级判定综合报告 (自适应降级模式)

> [!NOTE]
> 本报告由系统规则引擎自适应兜底生成，经过确定性质检标准核算，数据权威有效。

## 一、 生产批次与检测基本信息
- **生产批号**：{batch_id}
- **钢种牌号**：{steel_grade}
- **规格尺寸**：{batch_meta.get('specification', '-')}
- **生产产线**：{batch_meta.get('production_line', '-')}
- **受检缺陷总数**：{total_defects} 处

## 二、 质检统计核心数据
- **致命缺陷 (A类)**：{fatal_count} 处
- **严重缺陷 (B类)**：{severe_count} 处
- **轻微缺陷 (C类)**：{minor_count} 处
- **DQI 缺陷质量扣分**：{dqi:.1f} 分

## 三、 质量等级判定结论
- **最终质量等级**：【**{final_grade}**】
- **判定核心依据**：依据《钢卷表面及内部缺陷质量等级判定规范》，A类缺陷={fatal_count}，B类严重缺陷={severe_count}，C类轻微缺陷={minor_count}，综合质量扣分 DQI={dqi:.1f}。
- **质检处置建议**：{decision}。
"""

        return {
            "is_degraded": True,
            "final_markdown": degraded_markdown
        }

    def _route_validation(self, state: ReportWorkflowState) -> str:
        """条件边路由判断：状态机自适应重试与分支降级"""
        if state.get("parsed_report") is not None and state.get("consistency_passed", False):
            return "end"
        if state.get("retry_count", 0) <= state.get("max_retries", 2):
            return "retry"
        return "fallback"

    def _build_graph(self):
        """构建完整的 StateGraph 状态图"""
        builder = StateGraph(ReportWorkflowState)

        # 注册节点
        builder.add_node("sql_query", self._node_sql_query)
        builder.add_node("hybrid_rag", self._node_hybrid_rag)
        builder.add_node("generate_report", self._node_generate_report)
        builder.add_node("validate", self._node_validate_pydantic)
        builder.add_node("consistency_check", self._node_consistency_check)
        builder.add_node("retry", self._node_retry_with_feedback)
        builder.add_node("fallback", self._node_fallback_degradation)

        # 正常流水线流转
        builder.add_edge(START, "sql_query")
        builder.add_edge("sql_query", "hybrid_rag")
        builder.add_edge("hybrid_rag", "generate_report")
        builder.add_edge("generate_report", "validate")
        builder.add_edge("validate", "consistency_check")

        # 条件边跳转（一致性通过结束，不一致或校验失败触发重试，重试超限分支降级）
        builder.add_conditional_edges(
            "consistency_check",
            self._route_validation,
            {
                "end": END,
                "retry": "retry",
                "fallback": "fallback"
            }
        )
        builder.add_edge("retry", "validate")
        builder.add_edge("fallback", END)

        return builder.compile()

    def run_pipeline(self, batch_id: str, user_query: str = "") -> Dict[str, Any]:
        """运行完整状态机流水线并返回最终结果"""
        initial_state: ReportWorkflowState = {
            "batch_id": batch_id,
            "user_query": user_query or f"对批号 {batch_id} 进行质检评级",
            "sql_data": {},
            "sql_text": "",
            "rag_standards": "",
            "raw_llm_output": "",
            "parsed_report": None,
            "final_markdown": "",
            "consistency_passed": False,
            "consistency_error": None,
            "retry_count": 0,
            "max_retries": 2,
            "error_message": None,
            "is_degraded": False
        }

        final_state = self.graph.invoke(initial_state)
        return final_state


report_pipeline = SteelInspectionGraphPipeline()


if __name__ == '__main__':
    pipeline = SteelInspectionGraphPipeline()
    res = pipeline.run_pipeline("B202507-O5-01")
    print("最终 Markdown 报告:")
    print(res.get("final_markdown")[:500])
    print("是否触发降级:", res.get("is_degraded"))
