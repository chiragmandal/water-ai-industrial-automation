"""Basic tests for the industrial agent platform."""

from __future__ import annotations

import pathlib

import pytest

from mcp_server.tools.alert_trigger import clear_alerts, list_alerts, trigger_alert
from mcp_server.tools.sensor_query import query_sensor


@pytest.fixture(autouse=True)
def reset_alerts():
    clear_alerts()
    yield
    clear_alerts()


def test_query_sensor_normal_ranges():
    result = query_sensor(pump_id="pump-001", inject_anomaly=False)
    assert result["pump_id"] == "pump-001"
    assert "readings" in result
    readings = result["readings"]
    # Normal ranges should be loosely centred on training distributions.
    assert 50 < readings["temperature_c"] < 80
    assert 0.5 < readings["vibration_mm_s"] < 5
    assert 4 < readings["pressure_bar"] < 8
    assert 90 < readings["flow_rate_lpm"] < 150


def test_query_sensor_anomaly_injection():
    result = query_sensor(pump_id="pump-002", inject_anomaly=True)
    readings = result["readings"]
    # At least one reading should land in anomalous territory.
    high_temp = readings["temperature_c"] > 75
    high_vib = readings["vibration_mm_s"] > 4.5
    low_pressure = readings["pressure_bar"] < 5
    low_flow = readings["flow_rate_lpm"] < 100
    assert any([high_temp, high_vib, low_pressure, low_flow])


def test_anomaly_check_requires_trained_model():
    artifacts = (
        pathlib.Path(__file__).parent.parent / "artifacts" / "anomaly_model.joblib"
    )
    if not artifacts.exists():
        pytest.skip("Model not trained yet. Run `python -m mlops.train`.")

    from mcp_server.tools.anomaly_check import check_anomaly

    normal = check_anomaly(
        {
            "temperature_c": 65,
            "vibration_mm_s": 2.5,
            "pressure_bar": 6.0,
            "flow_rate_lpm": 120,
        }
    )
    assert "is_anomaly" in normal
    assert normal["is_anomaly"] is False

    anomalous = check_anomaly(
        {
            "temperature_c": 90,
            "vibration_mm_s": 9.0,
            "pressure_bar": 3.0,
            "flow_rate_lpm": 60,
        }
    )
    assert anomalous["is_anomaly"] is True
    assert anomalous["severity"] in {"warning", "critical"}


def test_trigger_alert_records_in_log():
    alert = trigger_alert(
        pump_id="pump-003",
        severity="critical",
        message="High vibration detected",
        recommended_action="shutdown_for_inspection",
    )
    assert alert["pump_id"] == "pump-003"
    assert alert["severity"] == "critical"
    assert alert["status"] == "open"
    assert any(a["alert_id"] == alert["alert_id"] for a in list_alerts())


def test_anomaly_check_missing_features():
    from mcp_server.tools.anomaly_check import check_anomaly

    result = check_anomaly({"temperature_c": 70})
    assert result["is_anomaly"] is False
    assert "error" in result
