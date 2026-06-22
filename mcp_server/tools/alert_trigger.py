"""Simulated alert and remediation system.

In production this would integrate with PagerDuty, ServiceNow, or
the plant's SCADA / CMMS to open work orders. Here we keep an
in-memory log so the workflow is observable end to end.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_ALERT_LOG: list[dict[str, Any]] = []


def trigger_alert(
    pump_id: str,
    severity: str,
    message: str,
    recommended_action: str = "inspect",
) -> dict[str, Any]:
    """Raise an alert for an anomalous pump."""
    alert = {
        "alert_id": f"alert-{len(_ALERT_LOG) + 1:05d}",
        "pump_id": pump_id,
        "severity": severity,
        "message": message,
        "recommended_action": recommended_action,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }
    _ALERT_LOG.append(alert)
    logger.warning(
        "ALERT %s pump=%s severity=%s msg=%s",
        alert["alert_id"],
        pump_id,
        severity,
        message,
    )
    return alert


def list_alerts() -> list[dict[str, Any]]:
    """Return all alerts raised in the current session."""
    return list(_ALERT_LOG)


def clear_alerts() -> None:
    """Reset the alert log. Used by tests."""
    _ALERT_LOG.clear()
