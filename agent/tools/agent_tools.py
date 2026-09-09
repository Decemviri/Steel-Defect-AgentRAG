import json
try:
    from langchain_core.tools import tool
except ImportError:
    try:
        from langchain.tools import tool
    except ImportError:
        def tool(*args, **kwargs):
            def decorator(func):
                func.description = kwargs.get("description", getattr(func, "__doc__", ""))
                func.invoke = lambda d: func(**d) if isinstance(d, dict) else func(d)
                return func
            if len(args) == 1 and callable(args[0]):
                return decorator(args[0])
            return decorator

from codes.logger_handler import logger
from rag.rag_service import RagSummaryService
from codes.sql_handler import (
    query_batch_defects_analysis,
    format_batch_defects_report_context,
    execute_sql,
    get_available_batches,
)

rag = RagSummaryService()


@tool(description="从向量知识库与BM25索引中检索钢铁缺陷分类、形貌特征、产生机理、工艺控制参数、消除对策与《钢卷表面及内部缺陷质量等级判定规范与判定标准手册》等专业资料")
def rag_summarize(query: str) -> str:
    return rag.generate_summary(query)


@tool(description="无入参，无返回值，调用后触发中间件自动为钢铁缺陷与质量报告生成场景动态注入上下文标志，实现提示词向专业质检报告模式平滑切换")
def fill_context_for_report():
    return "fill_context_for_report已调用"


@tool(description="调用 SQL 引擎与冶金智能质检处理流程查询指定生产批号（如'B202506-01'）或钢卷号，从底层缺陷数据库（类别、长、宽、卷号、长度位置、宽度位置、时间）提取真实缺陷明细，并动态研判严重度（A类致命/B类严重/C类轻微）、板面空间分布（边缘/中部/头尾）与形态（连续条状/密集片状/离散单点），输出标准结构化统计数据用于报告生成")
def query_batch_defects(batch_id: str) -> str:
    clean_batch = batch_id.strip().replace("'", "").replace('"', "")
    analysis = query_batch_defects_analysis(clean_batch)
    return format_batch_defects_report_context(analysis)


@tool(description="执行自定义只读 SQL 查询语句（针对缺陷表 defect_inspection/defect_inspection_cn 与批号表 batch_list），进行特定缺陷长宽尺寸统计、位置坐标过滤或多批次对比")
def execute_sql_query(query_sql: str) -> str:
    rows = execute_sql(query_sql)
    if not rows:
        return "查询结果为空"
    return str(rows[:20])


@tool(description="从质检数据库中获取所有可用的生产批号列表及钢种、规格与产线信息")
def get_batch_list() -> str:
    batches = get_available_batches()
    if not batches:
        return "当前数据库中暂无批号记录"
    lines = ["【系统中可用的生产批号清单】:"]
    for b in batches:
        lines.append(f"- 批号: {b['batch_id']} | 钢种: {b['steel_grade']} | 规格: {b['specification']} | 机组: {b['production_line']}")
    return "\n".join(lines)


@tool(description="使用 Schema RAG 动态注入包含缺陷类别、长、宽、卷号、长度位置、宽度位置与时间的表结构与业务术语，辅以 Few-shot 示例，将用户的任意自然语言查询自动转换为精准的 SQLite 查询并返回执行结果")
def natural_language_sql_query(user_question: str) -> str:
    from codes.schema_rag import schema_rag_engine
    res = schema_rag_engine.query_with_schema_rag(user_question)
    if "error" in res:
        return f"查询出错: {res['error']}"
    return f"【执行的 SQL】: {res.get('sql')}\n【查询结果条数】: {res.get('row_count')}\n【明细数据】: {json.dumps(res.get('results', [])[:15], ensure_ascii=False)}"


@tool(description="依托 LangGraph 状态机编排、Pydantic 强类型 JSON 输出约束、数据一致性核验与确定性分支降级机制，对指定批号（如'B202506-01'）一键生成高可靠性的《钢卷缺陷质检与质量等级判定综合报告》")
def generate_langgraph_inspection_report(batch_id: str) -> str:
    from agent.graph_pipeline import report_pipeline
    clean_batch = batch_id.strip().replace("'", "").replace('"', "")
    res = report_pipeline.run_pipeline(clean_batch)
    status_tag = "【自适应降级兜底生成】\n" if res.get("is_degraded") else "【Pydantic 强校验通过】\n"
    return status_tag + res.get("final_markdown", "未能生成报告")


if __name__ == '__main__':
    print(query_batch_defects.invoke({"batch_id": "B202506-01"}))
