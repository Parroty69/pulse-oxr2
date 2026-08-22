"""Hardware-aware deployment planning for the CXR co-pilot."""

from .hardware import DeploymentPlan, MachineProfile, detect_machine, select_deployment_plan

__all__ = [
    "DeploymentPlan",
    "MachineProfile",
    "detect_machine",
    "select_deployment_plan",
]
