"""Azure ML pipeline definition.

In production this submits a training job to an Azure ML compute
cluster using the v2 SDK. For local development the pipeline is
stubbed: if Azure credentials are not present, a placeholder response
is returned so the rest of the platform can be exercised without
incurring cloud costs.

To enable real Azure ML execution:
    pip install azure-ai-ml azure-identity
    az login
    export AZURE_SUBSCRIPTION_ID=...
    export AZURE_RESOURCE_GROUP=...
    export AZURE_ML_WORKSPACE=...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AzureMLConfig:
    subscription_id: str = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    resource_group: str = os.getenv("AZURE_RESOURCE_GROUP", "")
    workspace_name: str = os.getenv("AZURE_ML_WORKSPACE", "")
    compute_target: str = os.getenv("AZURE_COMPUTE", "cpu-cluster")

    @property
    def is_configured(self) -> bool:
        return all([self.subscription_id, self.resource_group, self.workspace_name])


def submit_training_job(config: AzureMLConfig | None = None) -> dict[str, Any]:
    """Submit the anomaly detection training job to Azure ML.

    Returns a stubbed response when Azure credentials are not configured.
    """
    config = config or AzureMLConfig()

    if not config.is_configured:
        logger.warning("Azure ML credentials not set, returning stubbed response.")
        return {
            "status": "stubbed",
            "job_name": "pump-anomaly-train-local",
            "message": (
                "Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP and "
                "AZURE_ML_WORKSPACE to submit a real Azure ML job."
            ),
        }

    try:
        from azure.ai.ml import MLClient, command
        from azure.ai.ml.entities import Environment
        from azure.identity import DefaultAzureCredential
    except ImportError:
        logger.error(
            "azure-ai-ml not installed. Run: pip install azure-ai-ml azure-identity"
        )
        return {"status": "error", "message": "azure-ai-ml not installed"}

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name,
    )

    job = command(
        code="./mlops",
        command="python train.py",
        environment=Environment(
            image="mcr.microsoft.com/azureml/curated/sklearn-1.1:latest"
        ),
        compute=config.compute_target,
        display_name="pump-anomaly-detection",
        experiment_name="industrial-agent-platform",
    )

    submitted = ml_client.jobs.create_or_update(job)
    return {
        "status": "submitted",
        "job_name": submitted.name,
        "studio_url": submitted.studio_url,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(submit_training_job())
