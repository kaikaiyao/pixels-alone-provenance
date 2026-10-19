from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_artifact_path(config: dict[str, Any], key: str) -> Path:
    override_root = os.environ.get("PP_ARTIFACTS_ROOT")
    configured = Path(config["paths"][key])
    if configured.is_absolute():
        return configured
    if override_root:
        artifacts_root = Path(override_root)
        if configured.parts and configured.parts[0] == "artifacts":
            configured = Path(*configured.parts[1:])
        return artifacts_root / configured
    workspace_root = Path(config["paths"].get("workspace_root", ".")).resolve()
    return (workspace_root / configured).resolve()


def resolve_artifact_relative(config: dict[str, Any], relative_path: str | os.PathLike[str]) -> Path:
    configured = Path(relative_path)
    if configured.is_absolute():
        return configured
    override_root = os.environ.get("PP_ARTIFACTS_ROOT")
    if override_root:
        artifacts_root = Path(override_root)
        if configured.parts and configured.parts[0] == "artifacts":
            configured = Path(*configured.parts[1:])
        return (artifacts_root / configured).resolve()
    workspace_root = Path(config["paths"].get("workspace_root", ".")).resolve()
    return (workspace_root / configured).resolve()


def resolve_workspace_path(config: dict[str, Any], relative_key: str) -> Path:
    root = Path(config["paths"].get("workspace_root", ".")).resolve()
    value = Path(config["paths"][relative_key])
    if value.is_absolute():
        return value
    return (root / value).resolve()


def resolve_workspace_relative(config: dict[str, Any], relative_path: str | os.PathLike[str]) -> Path:
    value = Path(relative_path)
    if value.is_absolute():
        return value
    root = Path(config["paths"].get("workspace_root", ".")).resolve()
    return (root / value).resolve()
