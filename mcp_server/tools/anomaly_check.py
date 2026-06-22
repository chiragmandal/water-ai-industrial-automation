"""Anomaly detection tool that calls the trained ML model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
FEATURES = ["temperature_c", "vibration_mm_s", "pressure_bar", "flow_rate_lpm"]

_model = None
_scaler = None


def _load_artifacts() -> tuple[Any, Any]:
    """Lazy load the model and scaler. Raises if training has not run."""
    global _model, _scaler
    if _model is None or _scaler is None:
        model_path = ARTIFACTS_DIR / "anomaly_model.joblib"
        scaler_path = ARTIFACTS_DIR / "scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {ARTIFACTS_DIR}. "
                "Run `python -m mlops.train` first."
            )
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        logger.info("Loaded anomaly model and scaler from %s", ARTIFACTS_DIR)
    return _model, _scaler


def check_anomaly(readings: dict[str, float]) -> dict[str, Any]:
    """Score sensor readings and return anomaly classification.

    Args:
        readings: Dict with keys temperature_c, vibration_mm_s,
            pressure_bar and flow_rate_lpm.
    """
    missing = [f for f in FEATURES if f not in readings]
    if missing:
        return {"is_anomaly": False, "error": f"Missing features: {missing}"}

    model, scaler = _load_artifacts()

    x = np.array([[readings[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)

    pred = int(model.predict(x_scaled)[0])
    score = float(model.score_samples(x_scaled)[0])

    is_anomaly = pred == -1
    # Severity must derive from is_anomaly to stay consistent. A point can
    # have a borderline score but be classed as an inlier; that's not an alert.
    if not is_anomaly:
        severity = "normal"
    elif score < -0.2:
        severity = "critical"
    else:
        severity = "warning"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(score, 4),
        "severity": severity,
        "features_evaluated": FEATURES,
    }
