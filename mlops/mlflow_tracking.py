"""MLflow tracking configuration helpers."""
from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def configure_mlflow(tracking_uri: str | None = None) -> None:
    """Set the MLflow tracking URI.

    Falls back to a local ./mlruns directory if no tracking server is
    available, so training works out of the box without docker compose.
    """
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)


def get_latest_run_id(experiment_name: str = "pump-anomaly-detection") -> str | None:
    """Return the most recent run id for an experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None
