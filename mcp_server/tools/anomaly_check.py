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

_pipeline = None


def _load_artifacts():
    """Lazy load the pipeline. Raises if training has not run."""
    global _pipeline
    if _pipeline is None:
        path = ARTIFACTS_DIR / "anomaly_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline artifact not found at {path}. "
                "Run `python -m mlops.train` first."
            )
        _pipeline = joblib.load(path)
        logger.info("Loaded anomaly pipeline from %s", path)
    return _pipeline


def check_anomaly(readings: dict[str, float]) -> dict[str, Any]:
    """Score sensor readings and return anomaly classification.

    Args:
        readings: Dict with keys temperature_c, vibration_mm_s,
            pressure_bar and flow_rate_lpm.
    """
    missing = [f for f in FEATURES if f not in readings]
    if missing:
        return {"is_anomaly": False, "error": f"Missing features: {missing}"}

    pipeline = _load_artifacts()

    x = np.array([[readings[f] for f in FEATURES]])

    pred = int(pipeline.predict(x)[0])
    score = float(pipeline.named_steps["iforest"].score_samples(
        pipeline.named_steps["scaler"].transform(x)
    )[0])

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
