import csv
import io
import json
import streamlit as st

from agent.react_agent import ReactAgent
from agent.graph_pipeline import report_pipeline
from codes.sql_handler import (
    get_available_batches,
    query_batch_defects_analysis,
    import_custom_csv_data,
    init_database,
)
from codes.schema_rag import schema_rag_engine

# =========================================================================
# 兼容性工具：兼容 Python 3.6 ~ Python 3.14 及 Streamlit 新旧 API
# =========================================================================
def render_dataframe(data):
    """
    自适应 Streamlit 新旧版本的数据表格渲染器。
    Streamlit 1.40+ 推荐 width='stretch'，旧版本使用 use_container_width=True。
    """
    try:
        st.dataframe(data, width="stretch")
    except (TypeError, Exception):
        try:
            st.dataframe(data, use_container_width=True)
        except Exception:
            st.dataframe(data)

# =========================================================================
# 页面基础配置与高质感工业 UI 样式系统
# =========================================================================
st.set_page_config(
    page_title="Steel-Defect-AgentRAG: 工业级钢卷缺陷智能质检与评级系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }

    /* 顶部高质感工业级大标题卡片 */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 6px;
    }
    .hero-tag {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        color: #e2e8f0;
        font-weight: 500;
    }

    /* 现代选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569;
        padding: 0 20px;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    /* KPI 卡片与指标仪表板 */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: #94a3b8;
    }
    .kpi-card.blue::before { background: #2563eb; }
    .kpi-card.red::before { background: #ef4444; }
    .kpi-card.amber::before { background: #f59e0b; }
    .kpi-card.emerald::before { background: #10b981; }
    .kpi-card.purple::before { background: #8b5cf6; }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }
    .kpi-num {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* 批次元数据卡片 */
    .batch-header-card {
        background: linear-gradient(to right, #f8fafc, #ffffff);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
    }
    .batch-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .meta-badge {
        display: inline-flex;
        align-items: center;
        background: #f1f5f9;
        color: #334155;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.84rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }

    /* 状态徽章 */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-pill.success {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .badge-pill.warning {
        background: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .badge-pill.danger {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* 侧边栏样式定制 */
    .sidebar-block-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* 聊天气泡美化 */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 会话状态初始化
# =========================================================================
if 'agent' not in st.session_state:
    st.session_state.agent = ReactAgent()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'auto_question' not in st.session_state:
    st.session_state.auto_question = ""

if 'current_report' not in st.session_state:
    st.session_state.current_report = None

if 'selected_batch_id' not in st.session_state:
    st.session_state.selected_batch_id = "B202506-01"

if 'nl_sql_query_input' not in st.session_state:
    st.session_state.nl_sql_query_input = ""

if 'last_uploaded_file_sig' not in st.session_state:
    st.session_state.last_uploaded_file_sig = None

# =========================================================================
# 顶部 Hero Banner
# =========================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>🏭 Steel-Defect-AgentRAG: 工业级钢卷缺陷智能诊断与质检评级专家系统</span>
    </div>
    <div class="hero-subtitle">
        基于空间坐标与冶金机理的多维研判 | 混合知识检索 (BM25 + Dense + BGE-Reranker) | Schema RAG 动态 SQL | LangGraph 容错状态机与一致性校验
    </div>
    <div class="hero-tags">
        <span class="hero-tag">🎯 缺陷数据库核心字段：类别 · 长 · 宽 · 卷号 · 长度位置 · 宽度位置 · 时间</span>
        <span class="hero-tag">📐 动态评定：空间分布(边缘/核心/头尾) + 聚集形态(连续/密集/离散)</span>
        <span class="hero-tag">🛡️ 可靠性闭环：Pydantic with_structured_output + 一致性校验杜绝幻觉 + 状态机降级</span>
        <span class="hero-tag">📜 国家标准五级评定：特级品 / 一级品 / 二级品 / 协议品 / 废品</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 获取数据库可用批号与元数据
available_batches = get_available_batches()
batch_map = {b["batch_id"]: b for b in available_batches}
batch_keys = [b["batch_id"] for b in available_batches]

# =========================================================================
# 侧边栏：全局控制台与配置
# =========================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-block-title">🎯 当前受检批次选择</div>', unsafe_allow_html=True)
    
    selected_idx = 0
    if st.session_state.selected_batch_id in batch_keys:
        selected_idx = batch_keys.index(st.session_state.selected_batch_id)
        
    chosen_batch = st.selectbox(
        "选择生产批号:",
        batch_keys,
        index=selected_idx,
        format_func=lambda bid: f"{bid} | {batch_map.get(bid, {}).get('steel_grade', '钢卷')}",
        key="sidebar_batch_select",
        help="选定批次后，系统将自动加载该批次下所有钢卷的真实缺陷长宽与位置数据"
    )
    if chosen_batch != st.session_state.selected_batch_id:
        st.session_state.selected_batch_id = chosen_batch
        st.session_state.current_report = None
        st.rerun()

    # 当前批号规格摘要卡
    curr_meta = batch_map.get(st.session_state.selected_batch_id, {})
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:0.75rem 1rem;margin-top:0.4rem;font-size:0.84rem;">
        <div style="color:#64748b;font-weight:600;margin-bottom:4px;">批次档案概要</div>
        <div>🏷️ <b>钢种牌号</b>: {curr_meta.get('steel_grade', '-')}</div>
        <div>📏 <b>规格尺寸</b>: {curr_meta.get('specification', '-')}</div>
        <div>⚙️ <b>生产机组</b>: {curr_meta.get('production_line', '-')}</div>
        <div>📅 <b>生产日期</b>: {curr_meta.get('production_date', '-')}</div>
        <div>🏢 <b>目标客户</b>: {curr_meta.get('target_customer', '-')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-block-title">⚙️ 报告生成工作流模式</div>', unsafe_allow_html=True)
    workflow_mode = st.radio(
        "选择报告编排引擎:",
        ["LangGraph 强约束流水线 (推荐)", "ReAct 智能体多轮协同"],
        index=0,
        help="LangGraph 模式集成 Pydantic 严格校验、错误反哺重试与规则兜底降级；ReAct 模式支持自由工具多步思考。"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-block-title">📂 数据导入与管理</div>', unsafe_allow_html=True)
    with st.expander("上传自定义批次 / 缺陷 CSV", expanded=False):
        uploaded_file = st.file_uploader(
            "选择 CSV 文件",
            type=["csv"],
            key="csv_file_uploader",
            help="支持中英文表头 (如包含 缺陷类别、长、宽、卷号、长度位置、宽度位置、时间 等)"
        )
        if uploaded_file is not None:
            # 严格防止由于重复执行导致的无限 rerun 陷阱
            file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_uploaded_file_sig != file_sig:
                st.session_state.last_uploaded_file_sig = file_sig
                try:
                    # 使用 utf-8-sig 兼容 Excel 导出的 BOM 头
                    content = uploaded_file.getvalue().decode("utf-8-sig")
                    reader = csv.DictReader(io.StringIO(content))
                    rows = list(reader)
                    if rows:
                        import_res = None
                        if any(k in rows[0] for k in ["defect_type", "缺陷类别", "缺陷的类别", "卷号", "coil_id", "length", "长"]):
                            import_res = import_custom_csv_data(defect_rows=rows)
                            st.success(f"✅ 成功导入 {len(rows)} 条缺陷质检数据！")
                        elif any(k in rows[0] for k in ["coil_count", "target_customer", "batch_id", "生产批号"]):
                            import_res = import_custom_csv_data(batch_rows=rows)
                            st.success(f"✅ 成功导入 {len(rows)} 条批号数据！")
                        
                        if import_res and import_res.get("active_batch_id"):
                            st.session_state.selected_batch_id = import_res["active_batch_id"]
                        st.session_state.current_report = None
                        st.rerun()
                except Exception as e:
                    st.error(f"解析上传 CSV 失败: {str(e)}")

        if st.button("🔄 重新初始化官方演示库", use_container_width=True):
            init_database(force_reload=True)
            st.session_state.last_uploaded_file_sig = None
            st.session_state.current_report = None
            st.success("已重载官方测试数据库！")
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ 清空对话历史与报告缓存", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_report = None
        st.rerun()

# 实时查询当前选中批次的缺陷与智能分析数据
batch_analysis = query_batch_defects_analysis(st.session_state.selected_batch_id)

# =========================================================================
# 主操作界面：四大核心功能模块选项卡
# =========================================================================
tab_overview, tab_report, tab_chat, tab_sql = st.tabs([
    "📊 批次缺陷全貌与质检数据库",
    "📑 综合质检评级与诊断报告",
    "💬 钢铁缺陷专家智能问答",
    "⚡ Schema RAG Text-to-SQL 探索"
])

# -------------------------------------------------------------------------
# TAB 1: 批次缺陷全貌与质检数据库
# -------------------------------------------------------------------------
with tab_overview:
    b_meta = batch_analysis.get("batch_meta", {})
    col_meta, col_act = st.columns([3, 1])
    
    with col_meta:
        st.markdown(f"""
        <div class="batch-header-card">
            <div>
                <div class="batch-title">📦 受检批次: {batch_analysis['batch_id']}</div>
                <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;">
                    <span class="meta-badge">钢种: {b_meta.get('steel_grade', '常规')}</span>
                    <span class="meta-badge">规格: {b_meta.get('specification', '标准')}</span>
                    <span class="meta-badge">产线: {b_meta.get('production_line', '主产线')}</span>
                    <span class="meta-badge">客户: {b_meta.get('target_customer', '通用')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_act:
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        btn_quick_report = st.button("📑 一键生成质检报告", type="primary", use_container_width=True, key="btn_quick_report")

    # 执行报告生成事件
    if btn_quick_report:
        with st.spinner(f"正在对批次 {st.session_state.selected_batch_id} 执行质检评级与报告生成..."):
            report_res = report_pipeline.run_pipeline(st.session_state.selected_batch_id)
            st.session_state.current_report = report_res
            st.success("✅ 质检评级报告已生成！已在下方及报告专区同步展示。")

    # KPI 数据指标行
    sev_map = {s["defect_severity"]: s["total_count"] for s in batch_analysis.get("severity_stats", [])}
    fatal_cnt = sev_map.get("A类致命缺陷", 0)
    severe_cnt = sev_map.get("B类严重缺陷", 0)
    minor_cnt = sev_map.get("C类轻微缺陷", 0)
    
    pos_map = {p["distribution_position"]: p["count"] for p in batch_analysis.get("position_stats", [])}
    edge_cnt = pos_map.get("边缘区", 0)
    core_cnt = pos_map.get("中部核心区", 0)
    head_tail_cnt = pos_map.get("卷头过渡区", 0)

    kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
    with kcol1:
        st.markdown(f"""
        <div class="kpi-card blue">
            <div class="kpi-label">受检缺陷总数</div>
            <div class="kpi-num">{batch_analysis['total_defects']} <span style="font-size:1rem;color:#64748b;font-weight:normal;">处</span></div>
            <div class="kpi-sub">涉及 {len(set(r.get('coil_id') for r in batch_analysis.get('raw_records', [])))} 卷钢卷</div>
        </div>
        """, unsafe_allow_html=True)
    with kcol2:
        st.markdown(f"""
        <div class="kpi-card red">
            <div class="kpi-label">A类致命缺陷 (一票否决)</div>
            <div class="kpi-num" style="color:#b91c1c;">{fatal_cnt} <span style="font-size:1rem;color:#64748b;font-weight:normal;">处</span></div>
            <div class="kpi-sub">{"🚨 触发一票否决判废" if fatal_cnt > 0 else "✅ 无致命缺陷"}</div>
        </div>
        """, unsafe_allow_html=True)
    with kcol3:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-label">B类严重缺陷 (限值受控)</div>
            <div class="kpi-num" style="color:#b45309;">{severe_cnt} <span style="font-size:1rem;color:#64748b;font-weight:normal;">处</span></div>
            <div class="kpi-sub">深划伤/氧化皮/折叠/大尺寸</div>
        </div>
        """, unsafe_allow_html=True)
    with kcol4:
        st.markdown(f"""
        <div class="kpi-card emerald">
            <div class="kpi-label">C类轻微缺陷 (外观级)</div>
            <div class="kpi-num" style="color:#15803d;">{minor_cnt} <span style="font-size:1rem;color:#64748b;font-weight:normal;">处</span></div>
            <div class="kpi-sub">轻微擦伤/微小水斑/油斑</div>
        </div>
        """, unsafe_allow_html=True)
    with kcol5:
        st.markdown(f"""
        <div class="kpi-card purple">
            <div class="kpi-label">空间分布研判</div>
            <div class="kpi-num" style="font-size:1.15rem;margin-top:8px;">
                边缘:<b>{edge_cnt}</b> | 中部:<b>{core_cnt}</b>
            </div>
            <div class="kpi-sub">卷头尾过渡段: {head_tail_cnt} 处</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 缺陷数据双视角切换展示
    data_view = st.radio("选择质检数据展示维度:", ["📋 缺陷数据库底层明细表 (类别/长/宽/卷号/长度位置/宽度位置/时间)", "🔍 冶金智能聚合研判表 (种类/尺寸/空间/形态/受累钢卷)"], horizontal=True)

    if "底层明细表" in data_view:
        st.caption("展示 SQLite 数据库真实存储结构（7大核心字段），包含根据实际几何尺寸与坐标动态推导的研判等级与分布：")
        raw_list = []
        for r in batch_analysis.get("raw_records", []):
            raw_list.append({
                "记录ID": r.get("id"),
                "卷号": r.get("coil_id"),
                "缺陷类别": r.get("defect_type"),
                "长 (mm)": r.get("length"),
                "宽 (mm)": r.get("width"),
                "长度位置 (m)": r.get("length_position"),
                "宽度位置 (mm)": r.get("width_position"),
                "检测时间": r.get("inspection_time"),
                "动态研判等级": r.get("defect_severity"),
                "空间分布区域": r.get("distribution_position"),
                "分布形态": r.get("distribution_pattern")
            })
        if raw_list:
            render_dataframe(raw_list)
        else:
            st.info("当前批次未检测到任何缺陷记录。")
    else:
        st.caption("按缺陷类别与严重级别进行聚合统计，反映各缺陷在受检钢卷上的发生频度与几何尺寸极值：")
        type_list = []
        for t in batch_analysis.get("type_stats", []):
            type_list.append({
                "缺陷类别": t.get("defect_type"),
                "研判严重等级": t.get("defect_severity"),
                "发生数量 (处)": t.get("count"),
                "最大长宽 (mm)": f"{t.get('max_length')} × {t.get('max_width')}",
                "空间分布区域": t.get("positions"),
                "分布形态": t.get("patterns"),
                "受影响钢卷编号": t.get("affected_coils")
            })
        if type_list:
            render_dataframe(type_list)
        else:
            st.info("暂无聚合数据。")

    # 快捷报告展开预览（就地直观呈现报告，无需强制切换页面）
    if st.session_state.current_report and st.session_state.current_report.get("batch_id") == st.session_state.selected_batch_id:
        with st.expander(f"📑 生产批号 {st.session_state.selected_batch_id} - 综合质检评级报告全文直览", expanded=True):
            r_data = st.session_state.current_report
            if r_data.get("is_degraded"):
                st.markdown(f'<div class="badge-pill warning">⚠️ 触发规则引擎自适应降级兜底生成 (重试 {r_data.get("retry_count", 0)} 次)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-pill success">✅ LangGraph + Pydantic Schema 强类型校验通过 (重试 {r_data.get("retry_count", 0)} 次)</div>', unsafe_allow_html=True)
            
            st.markdown(r_data.get("final_markdown", ""))
            st.download_button(
                label="💾 下载该批次报告 (.md)",
                data=r_data.get("final_markdown", ""),
                file_name=f"质检评级报告_{st.session_state.selected_batch_id}.md",
                mime="text/markdown",
                key="dl_btn_tab1"
            )

# -------------------------------------------------------------------------
# TAB 2: 综合质检评级与诊断报告
# -------------------------------------------------------------------------
with tab_report:
    st.markdown("### 📑 钢卷质量等级判定与综合质检报告")
    
    rep_col1, rep_col2 = st.columns([3, 1])
    with rep_col1:
        st.markdown(f"**受检对象**：生产批号 `{st.session_state.selected_batch_id}` | **执行引擎**：`{workflow_mode}`")
    with rep_col2:
        btn_run_report = st.button("🚀 重新生成报告", type="primary", use_container_width=True, key="btn_run_report")

    if btn_run_report:
        with st.spinner("LangGraph 状态机正在编排运行：SQL 提取 -> 混合标准检索 -> Pydantic 强校验 -> 生成报告..."):
            res = report_pipeline.run_pipeline(st.session_state.selected_batch_id)
            st.session_state.current_report = res
            st.success("报告生成完毕！")

    # 报告呈现区
    if st.session_state.current_report:
        report_data = st.session_state.current_report
        is_deg = report_data.get("is_degraded", False)
        retries = report_data.get("retry_count", 0)
        
        if is_deg:
            st.markdown(f'<div class="badge-pill warning">⚠️ 触发规则引擎自适应降级兜底生成 (重试 {retries} 次后分支降级)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-pill success">✅ LangGraph + Pydantic Schema 强类型校验通过 (重试 {retries} 次)</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        report_text = report_data.get("final_markdown", "报告内容为空")
        st.markdown(report_text)
        
        st.download_button(
            label="💾 下载质检分析报告 (.md)",
            data=report_text,
            file_name=f"质检评级报告_{st.session_state.selected_batch_id}.md",
            mime="text/markdown",
            key="dl_btn_tab2"
        )
    else:
        st.info("💡 尚未生成当前批次的质检评级报告。点击上方【🚀 重新生成报告】或前往第一页点击【一键生成质检报告】即可即时生成。")

# -------------------------------------------------------------------------
# TAB 3: 钢铁缺陷专家智能问答
# -------------------------------------------------------------------------
with tab_chat:
    st.markdown("### 💬 钢铁冶金与质检专家在线会话")
    
    # 快捷问答 Chip 按钮
    st.markdown("<div style='color:#64748b;font-size:0.85rem;font-weight:600;margin-bottom:6px;'>⚡ 常见缺陷咨询 & 典型批次快速诊断：</div>", unsafe_allow_html=True)
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    with chip_col1:
        if st.button("🚗 汽车外板 B202507-O5-01 特级品评定", use_container_width=True, key="chip_1"):
            st.session_state.auto_question = "针对汽车外板批号 B202507-O5-01 评定是否达到特级品"
            st.rerun()
        if st.button("🔍 冷轧表面划伤成因与消除方法", use_container_width=True, key="chip_2"):
            st.session_state.auto_question = "冷轧板表面划伤与划痕的主要成因及消除方法？"
            st.rerun()
            
    with chip_col2:
        if st.button("🏢 商用板 B202507-STD-02 质检报告", use_container_width=True, key="chip_3"):
            st.session_state.auto_question = "针对商用板批号 B202507-STD-02 生成质检评级报告"
            st.rerun()
        if st.button("🔬 连铸坯表面纵裂纹机理与控制", use_container_width=True, key="chip_4"):
            st.session_state.auto_question = "连铸坯表面纵裂纹的产生机理与工艺控制对策？"
            st.rerun()

    with chip_col3:
        if st.button("🚨 批号 B202507-REJ-05 贯穿裂纹判废", use_container_width=True, key="chip_5"):
            st.session_state.auto_question = "对批号 B202507-REJ-05 执行贯穿裂纹与严重夹杂判废"
            st.rerun()
        if st.button("📉 边缘划伤降为二级品原因分析", use_container_width=True, key="chip_6"):
            st.session_state.auto_question = "对批号 B202507-DEG-03 分析边缘划伤并判定二级品原因"
            st.rerun()

    st.markdown("---")

    # 对话历史气泡渲染
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👷" if msg["role"] == "user" else "🔬"):
            if msg.get("badge"):
                st.markdown(msg["badge"], unsafe_allow_html=True)
            st.markdown(msg["content"])

    def handle_user_chat(prompt_text):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user", avatar="👷"):
            st.markdown(prompt_text)
        
        with st.chat_message("assistant", avatar="🔬"):
            placeholder = st.empty()
            full_resp = ""
            with st.spinner("冶金专家正在检索知识库与数据库中..."):
                for chunk in st.session_state.agent.execute_stream(prompt_text):
                    full_resp += chunk
                    placeholder.markdown(full_resp + "▌")
            
            cleaned = full_resp.strip()
            if cleaned.lower().startswith(prompt_text.lower()):
                cleaned = cleaned[len(prompt_text):].strip()
            placeholder.markdown(cleaned)
        
        st.session_state.messages.append({"role": "assistant", "content": cleaned})

    # 处理输入或自动触发的问题
    if st.session_state.auto_question:
        q = st.session_state.auto_question
        st.session_state.auto_question = ""
        handle_user_chat(q)
    else:
        chat_val = st.chat_input("向钢铁缺陷专家提问，如：'分析冷轧板表面划伤产生机理' 或 '出具批号 B202506-02 报告'...")
        if chat_val:
            handle_user_chat(chat_val)

# -------------------------------------------------------------------------
# TAB 4: Schema RAG Text-to-SQL 探索
# -------------------------------------------------------------------------
with tab_sql:
    st.markdown("### ⚡ Schema RAG Text-to-SQL 智能查询控制台")
    st.caption("基于元数据动态注入、术语映射与 Few-shot 样本库，将自然语言提问自动编译为严格合法的 SQLite 查询。")
    
    with st.expander("📖 查看当前数据库表结构定义 (SQLite)", expanded=False):
        c_tbl1, c_tbl2 = st.columns(2)
        with c_tbl1:
            st.markdown("""
            **表 `defect_inspection` (缺陷明细表)**
            - `id` (INTEGER PRIMARY KEY)
            - `defect_type` (TEXT，缺陷类别)
            - `length` (REAL，缺陷延伸长度 mm)
            - `width` (REAL，缺陷横向宽度 mm)
            - `coil_id` (TEXT，卷号)
            - `length_position` (REAL，沿卷长位置 m)
            - `width_position` (REAL，沿板宽位置 mm)
            - `inspection_time` (TEXT，时间戳)
            """)
        with c_tbl2:
            st.markdown("""
            **表 `batch_list` (生产批号表)**
            - `batch_id` (TEXT PRIMARY KEY)
            - `steel_grade` (TEXT，钢种牌号)
            - `specification` (TEXT，厚度×宽度)
            - `coil_count` (INTEGER，钢卷总数)
            - `production_line` (TEXT，生产机组)
            - `production_date` (TEXT，生产日期)
            - `target_customer` (TEXT，目标客户)
            """)

    st.markdown("<div style='color:#64748b;font-size:0.85rem;font-weight:600;margin-bottom:6px;'>💡 快捷查询模版（点击自动填入）：</div>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("📊 统计各缺陷类别的发生数量与长宽极值", use_container_width=True, key="sql_tmpl_1"):
            st.session_state.nl_sql_query_input = "统计各缺陷类别的发生频次、平均长度与最大宽度"
    with q_col2:
        if st.button("🚨 查询沿轧向长度>=100mm的严重条带缺陷", use_container_width=True, key="sql_tmpl_2"):
            st.session_state.nl_sql_query_input = "查询沿轧向延伸长度超过100mm的严重条带状缺陷记录"
    with q_col3:
        if st.button("边缘板宽区域(<=50mm)的裂纹或划伤", use_container_width=True, key="sql_tmpl_3"):
            st.session_state.nl_sql_query_input = "查询板宽边缘区域(<=50mm)发生的所有划伤或裂纹明细"

    sql_input = st.text_input(
        "请输入您的自然语言查询需求:",
        value=st.session_state.nl_sql_query_input,
        placeholder="例如：查询检出贯穿裂纹或夹杂分层等致命缺陷的钢卷号、长宽与位置"
    )

    if st.button("🚀 运行 Text-to-SQL 分析", type="primary", use_container_width=True, key="btn_run_sql") and sql_input:
        with st.spinner("Schema RAG 正在匹配业务术语并生成安全 SQL..."):
            sql_res = schema_rag_engine.query_with_schema_rag(sql_input)
            if "error" in sql_res and sql_res.get("error"):
                st.error(sql_res["error"])
            else:
                st.markdown("**【生成的合法 SQLite 语句】**")
                st.code(sql_res.get("sql"), language="sql")
                st.caption(f"匹配返回记录数: {sql_res.get('row_count')} 条")
                
                results_data = sql_res.get("results", [])
                if results_data:
                    render_dataframe(results_data)
                else:
                    st.info("查询执行成功，但数据库中未检索到匹配的记录。")
