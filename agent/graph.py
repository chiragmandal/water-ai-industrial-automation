"""LangGraph agent orchestration for the industrial monitoring workflow.

The agent is a ReAct style loop with explicit tool use. It uses a
local Ollama model by default (free, no API key required). If Ollama
is unavailable or the model is not found, it falls back to Azure
OpenAI. Override via OLLAMA_MODEL / AZURE_OPENAI_* env vars.

The tools below wrap the same business logic that the MCP server
exposes, so the agent and external MCP clients stay in sync.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Annotated, AsyncGenerator, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agent.prompts import SYSTEM_PROMPT
from mcp_server.tools.alert_trigger import list_alerts, trigger_alert
from mcp_server.tools.anomaly_check import check_anomaly
from mcp_server.tools.sensor_query import query_sensor

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def query_sensor_tool(pump_id: str = "", inject_anomaly: bool = False) -> str:
    """Read current sensor readings for an industrial pump.

    Args:
        pump_id: Pump identifier (e.g. pump-001). Empty picks a random pump.
        inject_anomaly: Force anomalous readings, useful for testing.
    """
    result = query_sensor(pump_id=pump_id or None, inject_anomaly=inject_anomaly)
    return json.dumps(result)


@tool
def check_anomaly_tool(
    temperature_c: float,
    vibration_mm_s: float,
    pressure_bar: float,
    flow_rate_lpm: float,
) -> str:
    """Score sensor readings against the trained anomaly model.

    Returns severity (normal, warning, critical) and an anomaly score.
    """
    result = check_anomaly({
        "temperature_c": temperature_c,
        "vibration_mm_s": vibration_mm_s,
        "pressure_bar": pressure_bar,
        "flow_rate_lpm": flow_rate_lpm,
    })
    return json.dumps(result)


@tool
def trigger_alert_tool(
    pump_id: str,
    severity: str,
    message: str,
    recommended_action: str = "inspect",
) -> str:
    """Raise a maintenance alert for an anomalous pump."""
    result = trigger_alert(
        pump_id=pump_id,
        severity=severity,
        message=message,
        recommended_action=recommended_action,
    )
    return json.dumps(result)


@tool
def list_alerts_tool() -> str:
    """List all alerts raised in the current session."""
    return json.dumps({"alerts": list_alerts()})


TOOLS = [query_sensor_tool, check_anomaly_tool, trigger_alert_tool, list_alerts_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _build_ollama_llm():
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        timeout=30,
    )
    return llm.bind_tools(TOOLS).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )


def _build_azure_llm():
    missing = [
        v for v in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT")
        if not os.getenv(v)
    ]
    if missing:
        raise RuntimeError(f"Azure OpenAI fallback requires: {', '.join(missing)}")
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,
        timeout=60,
    )
    return llm.bind_tools(TOOLS).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )


_LLM = None
_GRAPH = None


def build_llm():
    """Try Ollama first; fall back to Azure OpenAI if Ollama is unreachable.

    Cached at module level so the smoke test runs once at startup, not on
    every graph step.
    """
    global _LLM
    if _LLM is not None:
        return _LLM
    try:
        llm = _build_ollama_llm()
        # Smoke-test the connection so failures surface here, not mid-graph.
        llm.invoke([HumanMessage(content="ping")])
        logger.info("Using Ollama (%s)", os.getenv("OLLAMA_MODEL", "llama3.1"))
        _LLM = llm
        return _LLM
    except Exception as ollama_err:
        logger.warning("Ollama unavailable (%s), falling back to Azure OpenAI", ollama_err)
        _LLM = _build_azure_llm()
        return _LLM


def get_graph():
    """Compiled graph singleton."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def agent_node(state: AgentState) -> dict:
    llm = build_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"messages": []}

    tool_messages: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        logger.info("agent_tool_call name=%s args=%s", name, args)
        if name not in TOOLS_BY_NAME:
            output = json.dumps({"error": f"Unknown tool {name}"})
        else:
            try:
                output = TOOLS_BY_NAME[name].invoke(args)
            except Exception as exc:
                logger.exception("Tool %s failed", name)
                output = json.dumps({"error": str(exc)})
        tool_messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def _initial_state(user_message: str) -> AgentState:
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    }


def run_agent(user_message: str) -> dict:
    """Run the agent on a single user message and return the final state."""
    return get_graph().invoke(
        _initial_state(user_message),
        config={"recursion_limit": 10},
    )


async def astream_agent(user_message: str) -> AsyncGenerator[dict, None]:
    """Stream agent events as they happen.

    Yields structured events suitable for SSE: tool_call, tool_result,
    final_answer. Used by the dashboard to render the agent loop live.
    """
    graph = get_graph()
    state = _initial_state(user_message)

    async for chunk in graph.astream(
        state,
        stream_mode="updates",
        config={"recursion_limit": 10},
    ):
        for node_name, payload in chunk.items():
            for msg in payload.get("messages", []):
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc["args"],
                                "call_id": tc.get("id", ""),
                            }
                    elif msg.content:
                        yield {
                            "type": "final_answer",
                            "content": str(msg.content),
                        }
                elif isinstance(msg, ToolMessage):
                    yield {
                        "type": "tool_result",
                        "call_id": getattr(msg, "tool_call_id", ""),
                        "content": str(msg.content),
                    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    state = run_agent("Check pump-002 for any issues right now.")
    for msg in state["messages"]:
        role = msg.__class__.__name__
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"\n=== {role} ===\n{content}")
