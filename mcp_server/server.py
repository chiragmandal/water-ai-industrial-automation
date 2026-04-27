"""MCP server exposing industrial monitoring tools to AI agents.

Run with:
    python -m mcp_server.server

This implements the Model Context Protocol over stdio. Any MCP
compatible client (Claude Desktop, Cursor, custom LangGraph clients,
etc.) can connect and discover the tools below.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_server.tools.alert_trigger import list_alerts, trigger_alert
from mcp_server.tools.anomaly_check import check_anomaly
from mcp_server.tools.sensor_query import query_sensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("mcp-server")

app: Server = Server("industrial-agent-platform")


@app.list_tools()
async def list_available_tools() -> list[Tool]:
    return [
        Tool(
            name="query_sensor",
            description=(
                "Read current sensor readings (temperature, vibration, "
                "pressure, flow) for an industrial pump."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pump_id": {
                        "type": "string",
                        "description": "Pump identifier such as pump-001. Omit to pick a random pump.",
                    },
                    "inject_anomaly": {
                        "type": "boolean",
                        "description": "Force anomalous readings for testing.",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="check_anomaly",
            description="Run the trained anomaly detection model on a set of sensor readings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "readings": {
                        "type": "object",
                        "description": "Sensor readings.",
                        "properties": {
                            "temperature_c": {"type": "number"},
                            "vibration_mm_s": {"type": "number"},
                            "pressure_bar": {"type": "number"},
                            "flow_rate_lpm": {"type": "number"},
                        },
                        "required": [
                            "temperature_c",
                            "vibration_mm_s",
                            "pressure_bar",
                            "flow_rate_lpm",
                        ],
                    },
                },
                "required": ["readings"],
            },
        ),
        Tool(
            name="trigger_alert",
            description="Raise a maintenance alert for an anomalous pump.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pump_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["warning", "critical"]},
                    "message": {"type": "string"},
                    "recommended_action": {"type": "string", "default": "inspect"},
                },
                "required": ["pump_id", "severity", "message"],
            },
        ),
        Tool(
            name="list_alerts",
            description="List all alerts triggered in the current session.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    logger.info("tool_call name=%s args=%s", name, arguments)

    if name == "query_sensor":
        result: Any = query_sensor(
            pump_id=arguments.get("pump_id"),
            inject_anomaly=arguments.get("inject_anomaly", False),
        )
    elif name == "check_anomaly":
        result = check_anomaly(arguments["readings"])
    elif name == "trigger_alert":
        result = trigger_alert(
            pump_id=arguments["pump_id"],
            severity=arguments["severity"],
            message=arguments["message"],
            recommended_action=arguments.get("recommended_action", "inspect"),
        )
    elif name == "list_alerts":
        result = {"alerts": list_alerts()}
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
