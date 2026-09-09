from typing import Callable, Any
from codes.logger_handler import logger

try:
    from langchain.agents import AgentState
    from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_community.utilities.pebblo import Runtime
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command
except ImportError:
    # 离线与兼容性装饰器兜底
    def wrap_tool_call(func):
        return func
    def before_model(func):
        return func
    def dynamic_prompt(func):
        return func
    AgentState = dict
    Runtime = Any
    ModelRequest = Any
    ToolCallRequest = Any
    ToolMessage = Any
    Command = Any


@wrap_tool_call
def monitor_tool(
        request: Any,
        handler: Callable[[Any], Any],
) -> Any:
    tool_name = getattr(request, "tool_call", {}).get("name", "unknown_tool") if hasattr(request, "tool_call") else "tool"
    tool_args = getattr(request, "tool_call", {}).get("args", {}) if hasattr(request, "tool_call") else {}
    logger.info(f"[Tool Monitor] 执行工具: {tool_name}")
    logger.info(f"[Tool Monitor] 传入参数: {tool_args}")

    try:
        result = handler(request)
        logger.info(f"[Tool Monitor] 工具 {tool_name} 调用成功")

        if hasattr(request, "tool_call") and request.tool_call.get("name") == "fill_context_for_report":
            if hasattr(request, "runtime") and request.runtime is not None:
                if request.runtime.context is None:
                    request.runtime.context = {}
                request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"[Tool Monitor] 工具 {tool_name} 调用异常: {str(e)}")
        raise e


@before_model
def log_before_model(
        state: Any,
        runtime: Any,
):
    """在模型执行前记录上下文状态"""
    if isinstance(state, dict) and "messages" in state:
        logger.info(f"[Prompt Middleware] 即将调用大模型，携带 {len(state['messages'])} 条消息")
        if state["messages"]:
            last_msg = state["messages"][-1]
            content = getattr(last_msg, "content", str(last_msg))
            logger.debug(f"[Prompt Middleware] 最新消息: {content[:100]}...")
    return None


@dynamic_prompt
def report_prompt_switch(request: Any):
    """根据运行时上下文状态动态切换质检专家系统提示词"""
    from codes.prompt_loader import load_report_prompts, load_system_prompts

    if not hasattr(request, "runtime") or request.runtime is None or request.runtime.context is None:
        return load_system_prompts()

    is_report = request.runtime.context.get("report", False)
    if is_report:
        logger.info("[Prompt Switch] 触发质检报告生成模式提示词切换")
        return load_report_prompts()

    return load_system_prompts()
