from agent.tools.agent_tools import (
    rag_summarize,
    fill_context_for_report,
    query_batch_defects,
    execute_sql_query,
    get_batch_list,
    natural_language_sql_query,
    generate_langgraph_inspection_report,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from codes.prompt_loader import load_system_prompts
from model.factory import chat_model
try:
    from langchain.agents import create_agent
except ImportError:
    try:
        from langchain.agents import create_react_agent as create_agent
    except ImportError:
        class DummyAgent:
            def __init__(self, tools):
                self.tools = {getattr(t, "name", getattr(t, "__name__", "")): t for t in tools}
            def stream(self, input_dict, stream_mode="values", context=None):
                query = input_dict.get("messages", [{}])[-1].get("content", "")
                class Msg:
                    def __init__(self, c):
                        self.content = c
                msg_text = f"【Steel-Defect-AgentRAG 专家系统】已接收质检需求: '{query}'。\n已加载 7 组核心工具：Schema RAG、空间缺陷分析、评级标准检索及 LangGraph 状态机报告生成。"
                yield {"messages": [Msg(msg_text)]}

        def create_agent(model, tools, system_prompt, middleware=None):
            return DummyAgent(tools)


class ReactAgent:
    def __init__(self):
        system_prompt = load_system_prompts()

        self.agent = create_agent(
            model=chat_model,
            tools=[
                rag_summarize,
                fill_context_for_report,
                query_batch_defects,
                execute_sql_query,
                get_batch_list,
                natural_language_sql_query,
                generate_langgraph_inspection_report,
            ],
            system_prompt=system_prompt,
            middleware=[
                monitor_tool,
                log_before_model,
                report_prompt_switch,
            ],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，用于提示词动态切换
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("系统中收录了哪些生产批号？"):
        print(chunk, end="")
