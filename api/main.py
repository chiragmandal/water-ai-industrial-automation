"""FastAPI wrapper around the industrial monitoring agent.

Exposes:
  GET  /health            healthcheck
  POST /agent/run         run the agent on a natural language instruction
  POST /sensor/query      simulated sensor read
  POST /anomaly/check     direct anomaly model scoring
  GET  /alerts            list raised alerts
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.graph import astream_agent, run_agent
from mcp_server.tools.alert_trigger import list_alerts
from mcp_server.tools.anomaly_check import check_anomaly
from mcp_server.tools.sensor_query import query_sensor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

import uuid
from contextvars import ContextVar

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id_var.get()
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s %(message)s",
    force=True,
)
for handler in logging.root.handlers:
    handler.addFilter(CorrelationIdFilter())

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting industrial-agent-platform API")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Industrial Agent Platform",
    description="AI agent for pump monitoring with MCP tool exposure.",
    version="0.1.0",
    lifespan=lifespan,
)

from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

from fastapi import Request


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    cid = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    token = correlation_id_var.set(cid)
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers["x-correlation-id"] = cid
    return response

class AgentRequest(BaseModel):
    message: str = Field(..., description="Natural language instruction for the agent.")


class AgentMessageDTO(BaseModel):
    role: str
    content: str
    tool_calls: list[dict] | None = None


class AgentResponse(BaseModel):
    final_answer: str
    trace: list[AgentMessageDTO]


class SensorRequest(BaseModel):
    pump_id: str | None = None
    inject_anomaly: bool = False


class AnomalyRequest(BaseModel):
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float
    flow_rate_lpm: float


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    """Serve the lightweight live dashboard."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return html_path.read_text(encoding="utf-8")


@app.post("/agent/stream")
async def agent_stream(req: AgentRequest) -> StreamingResponse:
    """Stream agent events as Server-Sent Events.

    The frontend dashboard consumes this to render tool calls and
    results in real time as the LangGraph executes.
    """
    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            async for event in astream_agent(req.message):
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            logger.exception("Agent stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/run", response_model=AgentResponse)
async def agent_run(req: AgentRequest) -> AgentResponse:
    try:
        state = run_agent(req.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    trace: list[AgentMessageDTO] = []
    final_answer = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            trace.append(AgentMessageDTO(role="user", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            tool_calls = [
                {"name": tc["name"], "args": tc["args"]}
                for tc in (msg.tool_calls or [])
            ]
            trace.append(AgentMessageDTO(
                role="assistant",
                content=str(msg.content) if msg.content else "",
                tool_calls=tool_calls or None,
            ))
            if msg.content and not msg.tool_calls:
                final_answer = str(msg.content)
        elif isinstance(msg, ToolMessage):
            trace.append(AgentMessageDTO(role="tool", content=str(msg.content)))

    return AgentResponse(final_answer=final_answer, trace=trace)


@app.post("/sensor/query")
async def sensor_query(req: SensorRequest) -> dict:
    return query_sensor(pump_id=req.pump_id, inject_anomaly=req.inject_anomaly)


@app.post("/anomaly/check")
async def anomaly_check(req: AnomalyRequest) -> dict:
    return check_anomaly(req.model_dump())


@app.get("/alerts")
async def alerts() -> dict:
    return {"alerts": list_alerts()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
