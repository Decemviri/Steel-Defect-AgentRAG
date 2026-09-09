from typing import List, Optional, Any, Dict
try:
    from typing import Literal
except ImportError:
    try:
        from typing_extensions import Literal
    except ImportError:
        class _Literal:
            def __getitem__(self, item):
                return Any
        Literal = _Literal()

try:
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        @classmethod
        def model_validate(cls, data: dict):
            if not isinstance(data, dict):
                raise ValueError("Expected dictionary data")
            return cls(**data)
        def model_dump(self) -> dict:
            return self.__dict__
        @classmethod
        def model_json_schema(cls) -> dict:
            return {"title": cls.__name__, "type": "object"}
    def Field(*args, **kwargs):
        return None
    def field_validator(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


if HAS_PYDANTIC:
    class DefectItem(BaseModel):
        """单项缺陷质检详情"""
        defect_type: str = Field(description="缺陷具体名称，如'表面裂纹'、'轻微擦伤'")
        severity: Literal["A类致命缺陷", "B类严重缺陷", "C类轻微缺陷"] = Field(description="严重级别")
        count: int = Field(ge=0, description="发生数量(处)")
        position: str = Field(description="分布位置，如'左侧边缘(<=50mm)'、'中部核心区'")
        pattern: str = Field(description="分布形态，如'离散单点'、'密集片状'、'连续条状'")
        size_mm: float = Field(ge=0.0, description="最大尺寸或受影响长度(mm)")
        length: Optional[float] = Field(default=None, description="缺陷长度(mm)")
        width: Optional[float] = Field(default=None, description="缺陷宽度(mm)")
        length_position: Optional[float] = Field(default=None, description="长度位置(m)")
        width_position: Optional[float] = Field(default=None, description="宽度位置(mm)")
        affected_coils: str = Field(description="受影响钢卷编号")

    class GradingEvaluation(BaseModel):
        """钢卷质量等级多维判定结论"""
        final_grade: Literal["特级品", "一级品", "二级品", "协议品", "废品"] = Field(
            description="最终判定质量等级"
        )
        dqi_score: float = Field(ge=0.0, description="综合质量缺陷扣分指数 (DQI)")
        grade_reason: str = Field(description="等级判定的核心依据，对照标准多维阐述")
        fatal_defects_count: int = Field(ge=0, description="A类致命缺陷总数")
        severe_defects_count: int = Field(ge=0, description="B类严重缺陷总数")
        minor_defects_count: int = Field(ge=0, description="C类轻微缺陷总数")
        position_compliance: str = Field(description="缺陷在边缘区(可切边)与核心中部(不可切)的符合性评估")
        is_rejected: bool = Field(description="是否判废(一票否决)")

        @field_validator("is_rejected")
        def validate_rejection(cls, v):
            return v

    class MetallurgicalAnalysis(BaseModel):
        """缺陷成因冶金学与工艺机理深度诊断"""
        primary_defect: str = Field(description="该批次最显著主导缺陷")
        process_stage: str = Field(description="诱发缺陷的主要工序阶段，如'连铸结晶器'、'热连轧精轧'、'酸洗冷连轧'")
        metallurgical_mechanism: str = Field(description="深度冶金学物理化学机理剖析")
        contributing_factors: List[str] = Field(description="主要诱发因素列表")

    class ActionPlan(BaseModel):
        """质检处置与产线改进对策"""
        disposition_decision: str = Field(description="质检出厂处置方案(正常放行/切边出库/降级使用/折价出库/返熔)")
        process_improvements: List[str] = Field(description="产线工艺优化与防范改进具体措施列表")

    class SteelInspectionReport(BaseModel):
        """
        钢铁缺陷质检与质量等级判定综合报告 (Pydantic 强约束模型)
        """
        batch_id: str = Field(description="受检生产批号")
        steel_grade: str = Field(description="钢种牌号")
        specification: str = Field(description="规格尺寸厚度×宽度")
        production_line: str = Field(description="生产机组产线")
        total_defects: int = Field(ge=0, description="缺陷累计总数")
        defects_summary: List[DefectItem] = Field(description="缺陷明细列表")
        grading: GradingEvaluation = Field(description="等级评定结论")
        metallurgical_analysis: MetallurgicalAnalysis = Field(description="成因机理诊断")
        action_plan: ActionPlan = Field(description="处置与改进建议")
        report_markdown: str = Field(description="符合规范的排版整齐的 Markdown 完整质检报告全文")

else:
    class DefectItem(BaseModel):
        pass

    class GradingEvaluation(BaseModel):
        pass

    class MetallurgicalAnalysis(BaseModel):
        pass

    class ActionPlan(BaseModel):
        pass

    class SteelInspectionReport(BaseModel):
        @classmethod
        def model_validate(cls, data: dict):
            if not isinstance(data, dict):
                raise ValueError("Expected dict")
            required_keys = ["batch_id", "steel_grade", "grading", "report_markdown"]
            for k in required_keys:
                if k not in data:
                    raise ValueError(f"Missing required field: {k}")
            grading = data.get("grading", {})
            if isinstance(grading, dict):
                fg = grading.get("final_grade")
                if fg not in ["特级品", "一级品", "二级品", "协议品", "废品"]:
                    raise ValueError(f"Invalid final_grade: {fg}")
            return cls(**data)

        @classmethod
        def model_json_schema(cls) -> dict:
            return {
                "title": "SteelInspectionReport",
                "type": "object",
                "required": ["batch_id", "steel_grade", "specification", "production_line", "total_defects", "grading", "report_markdown"],
                "properties": {
                    "batch_id": {"type": "string"},
                    "steel_grade": {"type": "string"},
                    "specification": {"type": "string"},
                    "production_line": {"type": "string"},
                    "total_defects": {"type": "integer"},
                    "defects_summary": {"type": "array"},
                    "grading": {
                        "type": "object",
                        "properties": {
                            "final_grade": {"type": "string", "enum": ["特级品", "一级品", "二级品", "协议品", "废品"]},
                            "dqi_score": {"type": "number"},
                            "grade_reason": {"type": "string"}
                        },
                        "required": ["final_grade", "dqi_score", "grade_reason"]
                    },
                    "report_markdown": {"type": "string"}
                }
            }
