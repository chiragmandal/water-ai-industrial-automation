"""Test the agent orchestration with a mock LLM.

Avoids real LLM calls in CI by feeding predetermined tool_call sequences.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def reset_state():
    from mcp_server.tools.alert_trigger import clear_alerts
    import agent.graph as g

    g._LLM = None
    g._GRAPH = None
    clear_alerts()
    yield
    clear_alerts()
    g._LLM = None
    g._GRAPH = None


class FakeLLM:
    """Returns a scripted sequence of AIMessages."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def invoke(self, messages, **kwargs):
        resp = self._responses[self._i]
        self._i += 1
        return resp

    def bind_tools(self, *args, **kwargs):
        return self

    def with_retry(self, *args, **kwargs):
        return self


def test_agent_calls_tools_in_correct_order():
    """Agent should query sensor -> check anomaly -> alert when severity is high."""
    artifacts = (
        __import__("pathlib").Path(__file__).parent.parent
        / "artifacts"
        / "anomaly_pipeline.joblib"
    )
    if not artifacts.exists():
        pytest.skip("Pipeline not trained")

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_sensor_tool",
                    "args": {"pump_id": "pump-001", "inject_anomaly": True},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_anomaly_tool",
                    "args": {
                        "temperature_c": 90,
                        "vibration_mm_s": 9.0,
                        "pressure_bar": 3.0,
                        "flow_rate_lpm": 60,
                    },
                    "id": "call_2",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "trigger_alert_tool",
                    "args": {
                        "pump_id": "pump-001",
                        "severity": "critical",
                        "message": "Test alert",
                    },
                    "id": "call_3",
                }
            ],
        ),
        AIMessage(content="Pump-001 has critical anomalies, alert raised."),
    ]

    fake = FakeLLM(scripted)
    with patch("agent.graph.build_llm", return_value=fake):
        import agent.graph as g

        g._GRAPH = None  # force rebuild with mock
        state = g.run_agent("Check pump-001")

    tool_calls = [
        m.tool_calls[0]["name"]
        for m in state["messages"]
        if isinstance(m, AIMessage) and m.tool_calls
    ]
    assert tool_calls == [
        "query_sensor_tool",
        "check_anomaly_tool",
        "trigger_alert_tool",
    ]


def test_agent_recursion_limit_enforced():
    """Agent must stop after N tool calls even if LLM keeps requesting more."""
    looping = [
        AIMessage(
            content="",
            tool_calls=[{"name": "list_alerts_tool", "args": {}, "id": f"call_{i}"}],
        )
        for i in range(50)
    ]
    fake = FakeLLM(looping)
    with patch("agent.graph.build_llm", return_value=fake):
        import agent.graph as g

        g._GRAPH = None
        from langgraph.errors import GraphRecursionError

        with pytest.raises(GraphRecursionError):
            g.run_agent("loop forever")
