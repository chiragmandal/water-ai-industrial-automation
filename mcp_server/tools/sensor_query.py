"""Simulated sensor data tool.

Generates realistic pump telemetry. In a real deployment this would
query an industrial historian like OSIsoft PI, an OPC UA server, or
a time-series database such as InfluxDB or TimescaleDB.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

PUMPS = ["pump-001", "pump-002", "pump-003", "pump-004"]


def query_sensor(
    pump_id: str | None = None, inject_anomaly: bool = False
) -> dict[str, Any]:
    """Return current sensor readings for an industrial pump.

    Args:
        pump_id: Specific pump identifier. When None, a pump is picked
            at random from the simulated fleet.
        inject_anomaly: When True, returns values drawn from the
            anomalous distribution. Useful for end to end testing.
    """
    pump_id = pump_id or random.choice(PUMPS)

    if inject_anomaly:
        readings = {
            "temperature_c": round(random.gauss(85, 6), 2),
            "vibration_mm_s": round(random.gauss(7.5, 1.2), 2),
            "pressure_bar": round(random.gauss(3.5, 0.8), 2),
            "flow_rate_lpm": round(random.gauss(70, 15), 2),
        }
    else:
        readings = {
            "temperature_c": round(random.gauss(65, 4), 2),
            "vibration_mm_s": round(random.gauss(2.5, 0.5), 2),
            "pressure_bar": round(random.gauss(6.0, 0.4), 2),
            "flow_rate_lpm": round(random.gauss(120, 8), 2),
        }

    return {
        "pump_id": pump_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "readings": readings,
        "unit_system": "metric",
    }
