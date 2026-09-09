import csv
import os
import re
import sqlite3
from typing import Dict, List, Any
from codes.logger_handler import logger
from codes.path_tool import get_abs_path


DB_PATH = get_abs_path("data/inspection.db")
DEFECT_CSV_PATH = get_abs_path("data/defect_inspection_database.csv")
BATCH_CSV_PATH = get_abs_path("data/batch_list.csv")
SAMPLE_DEFECT_CSV_PATH = get_abs_path("data/samples/sample_test_defects.csv")
SAMPLE_BATCH_CSV_PATH = get_abs_path("data/samples/sample_test_batches.csv")


def get_db_connection():
    """获取 SQLite 数据库连接"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(force_reload: bool = False):
    """
    初始化质检数据库与数据表，并从 CSV 载入数据。
    表结构包含用户指定的核心结构：
    - defect_type (缺陷的类别)
    - length (长，单位：mm)
    - width (宽，单位：mm)
    - coil_id (卷号)
    - length_position (长度位置，单位：m)
    - width_position (宽度位置，单位：mm)
    - inspection_time (时间)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if force_reload:
            cursor.execute("DROP VIEW IF EXISTS defect_inspection_cn")
            cursor.execute("DROP TABLE IF EXISTS defect_inspection")
            cursor.execute("DROP TABLE IF EXISTS batch_list")
            conn.commit()

        # 创建缺陷明细表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defect_inspection (
                id INTEGER PRIMARY KEY,
                defect_type TEXT,
                length REAL,
                width REAL,
                coil_id TEXT,
                length_position REAL,
                width_position REAL,
                inspection_time TEXT
            )
        """)

        # 创建批号索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_list (
                batch_id TEXT PRIMARY KEY,
                steel_grade TEXT,
                specification TEXT,
                coil_count INTEGER,
                production_line TEXT,
                production_date TEXT,
                target_customer TEXT
            )
        """)

        # 创建中文视图以便自然语言/中文 SQL 查询
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS defect_inspection_cn AS
            SELECT 
                id,
                defect_type AS 缺陷类别,
                defect_type AS 缺陷的类别,
                length AS 长,
                width AS 宽,
                coil_id AS 卷号,
                length_position AS 长度位置,
                width_position AS 宽度位置,
                inspection_time AS 时间
            FROM defect_inspection
        """)

        conn.commit()

        # 检查是否已有数据或强制重载
        cursor.execute("SELECT COUNT(*) FROM defect_inspection")
        defect_count = cursor.fetchone()[0]

        if defect_count == 0 or force_reload:
            cursor.execute("DELETE FROM defect_inspection")
            cursor.execute("DELETE FROM batch_list")
            conn.commit()

            def load_defect_file(file_path):
                if not os.path.exists(file_path):
                    return 0
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    records = []
                    for row in reader:
                        # 兼容中英文列名
                        rec_id = int(row.get("id") or len(records) + 1)
                        defect_type = (row.get("defect_type") or row.get("缺陷类别") or row.get("缺陷的类别") or "").strip()
                        length = float(row.get("length") or row.get("长") or 0.0)
                        width = float(row.get("width") or row.get("宽") or 0.0)
                        coil_id = (row.get("coil_id") or row.get("卷号") or "").strip()
                        length_position = float(row.get("length_position") or row.get("长度位置") or 0.0)
                        width_position = float(row.get("width_position") or row.get("宽度位置") or 0.0)
                        inspection_time = (row.get("inspection_time") or row.get("时间") or "").strip()

                        records.append((
                            rec_id,
                            defect_type,
                            length,
                            width,
                            coil_id,
                            length_position,
                            width_position,
                            inspection_time
                        ))
                    cursor.executemany("""
                        INSERT OR REPLACE INTO defect_inspection VALUES (?,?,?,?,?,?,?,?)
                    """, records)
                    return len(records)

            def load_batch_file(file_path):
                if not os.path.exists(file_path):
                    return 0
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    batches = []
                    for row in reader:
                        batches.append((
                            row["batch_id"].strip(),
                            row["steel_grade"].strip(),
                            row["specification"].strip(),
                            int(row["coil_count"]),
                            row["production_line"].strip(),
                            row["production_date"].strip(),
                            row["target_customer"].strip()
                        ))
                    cursor.executemany("""
                        INSERT OR REPLACE INTO batch_list VALUES (?,?,?,?,?,?,?)
                    """, batches)
                    return len(batches)

            c1 = load_defect_file(DEFECT_CSV_PATH)
            c2 = load_defect_file(SAMPLE_DEFECT_CSV_PATH)
            logger.info(f"成功导入 {c1 + c2} 条缺陷质检数据至 SQLite defect_inspection")

            b1 = load_batch_file(BATCH_CSV_PATH)
            b2 = load_batch_file(SAMPLE_BATCH_CSV_PATH)
            logger.info(f"成功导入 {b1 + b2} 条批号数据至 SQLite batch_list")

            conn.commit()
    except Exception as e:
        logger.error(f"初始化质检 SQLite 数据库异常: {str(e)}")
        raise e
    finally:
        conn.close()


def import_custom_csv_data(defect_rows: List[dict] = None, batch_rows: List[dict] = None) -> dict:
    """
    导入用户自定义上传的 CSV 批次或缺陷数据，支持中英文表头。
    自动关联补全 batch_list，并返回激活的批号ID供 UI 自动联动。
    """
    init_database()
    conn = get_db_connection()
    cursor = conn.cursor()
    active_batch = None
    imported_defects = 0
    imported_batches = 0

    try:
        if batch_rows:
            for b in batch_rows:
                bid = (b.get("batch_id") or b.get("生产批号") or "CUSTOM-BATCH").strip()
                if not active_batch:
                    active_batch = bid
                cursor.execute("""
                    INSERT OR REPLACE INTO batch_list VALUES (?,?,?,?,?,?,?)
                """, (
                    bid,
                    (b.get("steel_grade") or b.get("钢种牌号") or "自定义钢种").strip(),
                    (b.get("specification") or b.get("规格尺寸") or "1.2×1250mm").strip(),
                    int(b.get("coil_count") or b.get("钢卷总数") or 1),
                    (b.get("production_line") or b.get("生产产线") or "自定义产线").strip(),
                    (b.get("production_date") or b.get("生产日期") or "2025-08").strip(),
                    (b.get("target_customer") or b.get("目标客户") or "自定义客户").strip()
                ))
                imported_batches += 1

        if defect_rows:
            inferred_batches = {}
            cursor.execute("SELECT MAX(id) FROM defect_inspection")
            max_id_res = cursor.fetchone()
            curr_max_id = (max_id_res[0] if max_id_res and max_id_res[0] else 1000)

            for idx, d in enumerate(defect_rows):
                curr_max_id += 1
                try:
                    rec_id = int(d.get("id") or curr_max_id)
                except Exception:
                    rec_id = curr_max_id

                defect_type = (d.get("defect_type") or d.get("缺陷类别") or d.get("缺陷的类别") or "表面异常缺陷").strip()

                try:
                    length = float(d.get("length") or d.get("长") or 5.0)
                except Exception:
                    length = 5.0

                try:
                    width = float(d.get("width") or d.get("宽") or 2.0)
                except Exception:
                    width = 2.0

                raw_coil = (d.get("coil_id") or d.get("卷号") or "COIL-01").strip()
                raw_batch = (d.get("batch_id") or d.get("生产批号") or "").strip()

                if raw_batch:
                    batch_id = raw_batch
                    coil_id = raw_coil if raw_coil.startswith(raw_batch) else f"{raw_batch}-{raw_coil}"
                elif "-" in raw_coil:
                    batch_id = raw_coil.rsplit("-", 1)[0]
                    coil_id = raw_coil
                else:
                    batch_id = raw_coil
                    coil_id = raw_coil

                if not active_batch:
                    active_batch = batch_id

                if batch_id not in inferred_batches:
                    inferred_batches[batch_id] = {
                        "steel_grade": (d.get("steel_grade") or d.get("钢种牌号") or "导入受检钢种").strip(),
                        "specification": (d.get("specification") or d.get("规格尺寸") or "1.2×1250mm").strip()
                    }

                try:
                    length_position = float(d.get("length_position") or d.get("长度位置") or 10.0)
                except Exception:
                    length_position = 10.0

                try:
                    width_position = float(d.get("width_position") or d.get("宽度位置") or 50.0)
                except Exception:
                    width_position = 50.0

                inspection_time = (d.get("inspection_time") or d.get("时间") or "2025-08-01 10:00").strip()

                cursor.execute("""
                    INSERT OR REPLACE INTO defect_inspection VALUES (?,?,?,?,?,?,?,?)
                """, (
                    rec_id,
                    defect_type,
                    length,
                    width,
                    coil_id,
                    length_position,
                    width_position,
                    inspection_time
                ))
                imported_defects += 1

            for bid, binfo in inferred_batches.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO batch_list VALUES (?,?,?,?,?,?,?)
                """, (
                    bid,
                    binfo["steel_grade"],
                    binfo["specification"],
                    1,
                    "自定义导入机组",
                    "2025-08",
                    "自定义质检项目"
                ))

        conn.commit()
        logger.info(f"自定义 CSV 成功入库: 缺陷 {imported_defects} 条, 批次 {imported_batches} 个, 当前激活批次: {active_batch}")
        return {
            "success": True,
            "imported_defects": imported_defects,
            "imported_batches": imported_batches,
            "active_batch_id": active_batch
        }
    except Exception as e:
        logger.error(f"导入自定义 CSV 数据出错: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "active_batch_id": active_batch
        }
    finally:
        conn.close()


def execute_sql(sql_query: str) -> List[dict]:
    """执行通用只读 SQL 查询并返回列表字典"""
    init_database()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"执行 SQL 出错 [{sql_query}]: {str(e)}")
        return [{"error": str(e)}]
    finally:
        conn.close()


def get_available_batches() -> List[dict]:
    """获取所有可用批号列表"""
    init_database()
    return execute_sql("SELECT * FROM batch_list ORDER BY batch_id ASC")


# =========================================================================
# 冶金质检处理流程：基于长、宽、长度位置、宽度位置的动态研判算法
# =========================================================================

def parse_specification_width(spec_str: str) -> float:
    """从规格尺寸文本（如 '1.2×1250mm', '2.5×1500mm'）中解析板宽"""
    try:
        m = re.search(r'[×xX*](\d+(?:\.\d+)?)', spec_str)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 1250.0  # 默认基准板宽 1250mm


def eval_spatial_position(length_pos: float, width_pos: float, spec_width: float = 1250.0) -> dict:
    """
    根据长度位置(m)与宽度位置(mm)动态判定空间分布区域：
    - 卷头过渡区：length_pos <= 10.0m (轧制加减速段，切头切尾消除，影响权重 0.5)
    - 边缘区：距板两侧边部 <= 50mm (可通过后续切边工序消除，影响权重 0.6)
    - 中部核心区：距边部 > 50mm 的板面中心承载受力区 (严禁严重缺陷，影响权重 1.5)
    """
    if length_pos <= 10.0:
        return {
            "zone": f"卷头过渡区(0-10m, 实测{length_pos:.1f}m)",
            "zone_category": "卷头过渡区",
            "is_edge": False,
            "is_core": False,
            "is_head_tail": True,
            "weight_factor": 0.5
        }

    edge_threshold = 50.0
    if width_pos <= edge_threshold:
        return {
            "zone": f"左侧边缘区(<=50mm, 距边{width_pos:.1f}mm)",
            "zone_category": "边缘区",
            "is_edge": True,
            "is_core": False,
            "is_head_tail": False,
            "weight_factor": 0.6
        }
    elif spec_width > 0 and width_pos >= (spec_width - edge_threshold):
        dist_to_right = spec_width - width_pos
        return {
            "zone": f"右侧边缘区(<=50mm, 距边{dist_to_right:.1f}mm)",
            "zone_category": "边缘区",
            "is_edge": True,
            "is_core": False,
            "is_head_tail": False,
            "weight_factor": 0.6
        }
    else:
        return {
            "zone": f"中部核心区(板宽{width_pos:.1f}mm)",
            "zone_category": "中部核心区",
            "is_edge": False,
            "is_core": True,
            "is_head_tail": False,
            "weight_factor": 1.5
        }


def eval_defect_pattern(length: float, width: float, is_clustered: bool = False) -> str:
    """
    根据长、宽及密集度动态研判分布形态：
    - 连续条状：沿轧向延伸长度 length >= 300mm 的长尺寸划伤或纵裂
    - 密集片状：临近单位面积聚集多处
    - 离散单点：孤立独立点
    """
    if length >= 300.0:
        return "连续条状"
    if is_clustered:
        return "密集片状"
    return "离散单点"


def eval_defect_severity(defect_type: str, length: float, width: float, zone_cat: str) -> str:
    """
    依据冶金质检标准，根据缺陷类别与长宽尺寸动态研判严重等级 (A类致命 / B类严重 / C类轻微)：
    - A类致命缺陷：贯穿性裂纹、大颗粒夹杂分层、严重贯穿孔洞、严重结疤裂纹
    - B类严重缺陷：明显折叠、深划伤、重度氧化铁皮、边缘裂纹，或尺寸超标(长>=10mm或宽>=5mm)
    - C类轻微缺陷：轻微擦伤、微小水斑、油斑、微小麻点、轻微辊印
    """
    d_name = defect_type.strip()
    fatal_keywords = ["贯穿", "穿透", "穿孔", "夹杂分层", "致命"]
    if any(k in d_name for k in fatal_keywords):
        return "A类致命缺陷"
    if "裂纹" in d_name and (length >= 50.0 or zone_cat == "中部核心区"):
        return "A类致命缺陷"

    severe_keywords = ["折叠", "重皮", "氧化铁皮", "深划伤", "划痕", "边裂", "裂纹", "严重结疤", "气泡", "辊印"]
    if any(k in d_name for k in severe_keywords):
        return "B类严重缺陷"
    if length >= 10.0 or width >= 5.0:
        return "B类严重缺陷"

    return "C类轻微缺陷"


def query_batch_defects_analysis(query_id: str) -> Dict[str, Any]:
    """
    按批号或卷号执行深度 SQL 查询与冶金智能质检处理流程：
    1. 从 SQLite 中提取受检钢卷的所有真实缺陷记录 (类别、长、宽、卷号、长度位置、宽度位置、时间)；
    2. 基于实际板宽与坐标动态推导空间分布 (边缘/中部/头尾)；
    3. 基于缺陷长宽几何尺寸动态研判分布形态 (连续条状/密集片状/离散单点)；
    4. 对照标准动态确定缺陷严重度 (A/B/C) 与扣分影响；
    5. 聚合统计严重度、种类特征、受损钢卷及空间分布。
    """
    init_database()
    clean_id = query_id.strip()

    # 1. 获取批次基本元数据
    batch_info = execute_sql(f"SELECT * FROM batch_list WHERE batch_id = '{clean_id}'")
    if not batch_info or "error" in batch_info[0]:
        # 尝试通过卷号前缀逆向查找批号 (如 'B202506-01-01' -> 'B202506-01')
        prefix = clean_id.rsplit("-", 1)[0] if "-" in clean_id else clean_id
        batch_info = execute_sql(f"SELECT * FROM batch_list WHERE batch_id = '{prefix}'")

    batch_meta = batch_info[0] if batch_info and "error" not in batch_info[0] else {}
    spec_width = parse_specification_width(batch_meta.get("specification", "1250mm"))

    # 2. 查询对应的缺陷记录 (匹配批号或单卷)
    sql_query = f"""
        SELECT * FROM defect_inspection 
        WHERE coil_id LIKE '{clean_id}%' OR coil_id = '{clean_id}'
        ORDER BY id ASC
    """
    raw_rows = execute_sql(sql_query)
    if not raw_rows or ("error" in raw_rows[0]):
        raw_rows = []

    # 3. 动态处理流程：计算各记录的空间位置、聚集形态与严重级别
    processed_records = []
    # 密集度判定辅助：检查是否有临近聚集
    for i, r in enumerate(raw_rows):
        l_pos = float(r.get("length_position") or 0.0)
        w_pos = float(r.get("width_position") or 0.0)
        d_len = float(r.get("length") or 0.0)
        d_wid = float(r.get("width") or 0.0)
        d_type = str(r.get("defect_type") or "")
        coil = str(r.get("coil_id") or "")
        time_str = str(r.get("inspection_time") or "")

        pos_info = eval_spatial_position(l_pos, w_pos, spec_width)
        
        # 检查是否与同行同卷相邻缺陷聚集（3米长、100mm宽范围）
        is_clustered = False
        cluster_cnt = 0
        for j, other in enumerate(raw_rows):
            if i != j and other.get("coil_id") == coil:
                ol_pos = float(other.get("length_position") or 0.0)
                ow_pos = float(other.get("width_position") or 0.0)
                if abs(ol_pos - l_pos) <= 3.0 and abs(ow_pos - w_pos) <= 100.0:
                    cluster_cnt += 1
        if cluster_cnt >= 2:
            is_clustered = True

        pattern = eval_defect_pattern(d_len, d_wid, is_clustered)
        severity = eval_defect_severity(d_type, d_len, d_wid, pos_info["zone_category"])

        processed_records.append({
            "id": r.get("id"),
            "defect_type": d_type,
            "length": d_len,
            "width": d_wid,
            "coil_id": coil,
            "length_position": l_pos,
            "width_position": w_pos,
            "inspection_time": time_str,
            "distribution_position": pos_info["zone"],
            "zone_category": pos_info["zone_category"],
            "distribution_pattern": pattern,
            "defect_severity": severity,
            "weight_factor": pos_info["weight_factor"]
        })

    # 4. 聚合统计
    # (1) 严重度统计
    severity_counter = {}
    for pr in processed_records:
        sev = pr["defect_severity"]
        severity_counter[sev] = severity_counter.get(sev, 0) + 1

    severity_stats = []
    for sev_name in ["A类致命缺陷", "B类严重缺陷", "C类轻微缺陷"]:
        if sev_name in severity_counter:
            severity_stats.append({
                "defect_severity": sev_name,
                "total_count": severity_counter[sev_name]
            })

    # (2) 缺陷种类与空间分布聚合统计
    type_dict = {}
    for pr in processed_records:
        t_key = (pr["defect_type"], pr["defect_severity"])
        if t_key not in type_dict:
            type_dict[t_key] = {
                "defect_type": pr["defect_type"],
                "defect_severity": pr["defect_severity"],
                "count": 0,
                "positions": set(),
                "patterns": set(),
                "max_length": 0.0,
                "max_width": 0.0,
                "affected_coils": set(),
                "time_samples": set()
            }
        type_dict[t_key]["count"] += 1
        type_dict[t_key]["positions"].add(pr["distribution_position"])
        type_dict[t_key]["patterns"].add(pr["distribution_pattern"])
        type_dict[t_key]["max_length"] = max(type_dict[t_key]["max_length"], pr["length"])
        type_dict[t_key]["max_width"] = max(type_dict[t_key]["max_width"], pr["width"])
        type_dict[t_key]["affected_coils"].add(pr["coil_id"])
        if pr["inspection_time"]:
            type_dict[t_key]["time_samples"].add(pr["inspection_time"])

    type_stats = []
    for (d_type, sev), info in type_dict.items():
        type_stats.append({
            "defect_type": d_type,
            "defect_severity": sev,
            "count": info["count"],
            "positions": "；".join(list(info["positions"])),
            "patterns": "、".join(list(info["patterns"])),
            "max_length": info["max_length"],
            "max_width": info["max_width"],
            "max_size_mm": max(info["max_length"], info["max_width"]),
            "affected_coils": ", ".join(list(info["affected_coils"])),
            "time_samples": ", ".join(list(info["time_samples"])[:2])
        })
    type_stats.sort(key=lambda x: x["count"], reverse=True)

    # (3) 区域分布统计
    pos_counter = {}
    for pr in processed_records:
        z = pr["zone_category"]
        pos_counter[z] = pos_counter.get(z, 0) + 1

    position_stats = []
    for z_name, c in pos_counter.items():
        position_stats.append({
            "distribution_position": z_name,
            "count": c
        })
    position_stats.sort(key=lambda x: x["count"], reverse=True)

    total_defects = len(processed_records)

    return {
        "batch_id": clean_id,
        "batch_meta": batch_meta,
        "total_defects": total_defects,
        "severity_stats": severity_stats,
        "type_stats": type_stats,
        "position_stats": position_stats,
        "raw_records": processed_records
    }


def format_batch_defects_report_context(analysis: Dict[str, Any]) -> str:
    """将质检数据库分析结果格式化为供 LLM/Agent 诊断评级的标准专业文本"""
    if not analysis.get("raw_records"):
        return f"未能检索到生产批号或钢卷 {analysis.get('batch_id')} 的任何质检缺陷记录，请检查输入是否正确。"

    batch_meta = analysis.get("batch_meta", {})
    lines = [
        f"==== 生产批号/钢卷 {analysis.get('batch_id')} 质检数据库 SQL 查询与智能分析结果 ====",
        f"【钢种牌号】: {batch_meta.get('steel_grade', '未知')}",
        f"【规格尺寸】: {batch_meta.get('specification', '未知')}",
        f"【生产机组】: {batch_meta.get('production_line', '未知')}",
        f"【生产日期】: {batch_meta.get('production_date', '未知')}",
        f"【目标客户/用途】: {batch_meta.get('target_customer', '常规应用')}",
        f"【受检缺陷总数】: {analysis.get('total_defects')} 处",
        "",
        "--- 1. 缺陷严重级别 (Severity) 动态统计 ---"
    ]

    for s in analysis.get("severity_stats", []):
        lines.append(f"- {s.get('defect_severity')}: 共 {s.get('total_count')} 处")

    lines.append("")
    lines.append("--- 2. 缺陷种类、几何尺寸(长×宽)与空间分布明细 ---")
    for t in analysis.get("type_stats", []):
        lines.append(
            f"- 缺陷类别: {t.get('defect_type')} [{t.get('defect_severity')}] | 数量: {t.get('count')} 处 | "
            f"最大长宽: {t.get('max_length')}mm × {t.get('max_width')}mm | "
            f"分布区域: {t.get('positions')} | 形态: {t.get('patterns')} | "
            f"受累钢卷: {t.get('affected_coils')}"
        )

    lines.append("")
    lines.append("--- 3. 空间分布 (板宽边缘 vs 中部核心区 vs 卷头过渡区) 统计 ---")
    for p in analysis.get("position_stats", []):
        lines.append(f"- 分布区域 [{p.get('distribution_position')}]: 累计 {p.get('count')} 处")

    lines.append("")
    lines.append("--- 4. 底层缺陷明细数据样本 (前5条) ---")
    for r in analysis.get("raw_records", [])[:5]:
        lines.append(
            f"  * [ID={r.get('id')}] 卷号={r.get('coil_id')} | 类别={r.get('defect_type')} | "
            f"长={r.get('length')}mm, 宽={r.get('width')}mm | 长度位置={r.get('length_position')}m, 宽度位置={r.get('width_position')}mm | "
            f"时间={r.get('inspection_time')} -> 评定:{r.get('defect_severity')}({r.get('distribution_position')})"
        )

    lines.append("=========================================================")
    return "\n".join(lines)


if __name__ == '__main__':
    init_database(force_reload=True)
    batches = get_available_batches()
    print("可用批号:", [b["batch_id"] for b in batches])
    res = query_batch_defects_analysis("B202506-01")
    print(format_batch_defects_report_context(res))
