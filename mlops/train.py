"""Anomaly detection model training with MLflow tracking.

Trains an Isolation Forest on synthetic pump sensor data. The model
detects abnormal pump behaviour from temperature, vibration, pressure,
and flow rate readings.

Usage:
    python -m mlops.train
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlops.mlflow_tracking import configure_mlflow

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURES = ["temperature_c", "vibration_mm_s", "pressure_bar", "flow_rate_lpm"]


def generate_synthetic_pump_data(
    n_normal: int = 5000, n_anomaly: int = 250, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic pump telemetry with realistic ranges.

    Normal operation values are drawn from gaussian distributions
    centred on typical industrial pump operating points. Anomalies
    simulate cavitation, bearing failure, and partial blockage.
    """
    rng = np.random.default_rng(seed)

    normal = pd.DataFrame({
        "temperature_c": rng.normal(65, 4, n_normal),
        "vibration_mm_s": rng.normal(2.5, 0.5, n_normal),
        "pressure_bar": rng.normal(6.0, 0.4, n_normal),
        "flow_rate_lpm": rng.normal(120, 8, n_normal),
        "label": 0,
    })

    anomalies = pd.DataFrame({
        "temperature_c": rng.normal(85, 6, n_anomaly),
        "vibration_mm_s": rng.normal(7.5, 1.2, n_anomaly),
        "pressure_bar": rng.normal(3.5, 0.8, n_anomaly),
        "flow_rate_lpm": rng.normal(70, 15, n_anomaly),
        "label": 1,
    })

    df = pd.concat([normal, anomalies], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def train(contamination: float = 0.05, n_estimators: int = 200) -> dict:
    """Train the anomaly detector and log to MLflow."""
    configure_mlflow()
    mlflow.set_experiment("pump-anomaly-detection")

    with mlflow.start_run() as run:
        df = generate_synthetic_pump_data()
        X = df[FEATURES].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Pipeline keeps scaler and model versioned as one atomic artifact.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        # Unsupervised: fit on normal samples only.
        X_train_normal = X_train[y_train == 0]
        pipeline.fit(X_train_normal)

        preds = pipeline.predict(X_test)
        # Isolation Forest returns 1 for inliers and -1 for outliers.
        y_pred = (preds == -1).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "contamination": contamination,
            "n_train_samples": len(X_train_normal),
            "features": ",".join(FEATURES),
        })
        mlflow.log_metrics({
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
        })

        mlflow.sklearn.log_model(pipeline, artifact_path="anomaly_pipeline")

        # Persist locally so the MCP tool can load without MLflow at runtime.
        joblib.dump(pipeline, ARTIFACTS_DIR / "anomaly_pipeline.joblib")
        logger.info("Training complete. Run ID: %s", run.info.run_id)
        print(classification_report(y_test, y_pred))

        return {
            "run_id": run.info.run_id,
            "metrics": {
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"],
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
