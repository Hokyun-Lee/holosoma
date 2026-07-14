"""Optional Weights & Biases logging for motion-generator training.

The adapter deliberately has no top-level dependency on :mod:`wandb`.  A
disabled logger is therefore a true no-op even when wandb is not installed,
while online and offline modes import the package only when ``init`` is
called.  This keeps generator inference and existing Stage 1--8 presets free
from an optional training dependency.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

WandbMode = Literal["disabled", "online", "offline"]


@dataclass
class WandbLoggingConfig:
    """Configuration owned by the lightweight motion-generator adapter."""

    mode: WandbMode = "disabled"
    entity: str | None = None
    project: str = "HoloSomaMotionGenerator"
    name: str | None = None
    group: str | None = None
    notes: str | None = None
    run_id: str | None = None
    resume: bool | Literal["allow", "never", "must", "auto"] | None = None
    tags: list[str] = field(default_factory=list)
    directory: str | None = None
    job_type: str = "train"
    save_code: bool = True
    log_final_checkpoint_artifact: bool = False
    artifact_name: str | None = None


def flatten_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
    separator: str = "/",
) -> dict[str, Any]:
    """Flatten nested metric mappings without changing their leaf values.

    ``{"val": {"loss": 1.0}}`` becomes ``{"val/loss": 1.0}``.  A
    collision between an already-separated key and a nested key is rejected
    rather than silently overwriting one metric.
    """

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if not separator:
        raise ValueError("separator must not be empty")

    flattened: dict[str, Any] = {}

    def visit(values: Mapping[str, Any], parent: str) -> None:
        for raw_key, value in values.items():
            key_part = str(raw_key)
            key = f"{parent}{separator}{key_part}" if parent else key_part
            if isinstance(value, Mapping):
                visit(value, key)
            elif key in flattened:
                raise ValueError(f"flattened metric key collision: {key!r}")
            else:
                flattened[key] = value

    visit(metrics, prefix)
    return flattened


class MotionGenWandbLogger:
    """Small stateful adapter around a lazily imported wandb run."""

    def __init__(self, config: WandbLoggingConfig | None = None):
        self.config = config or WandbLoggingConfig()
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._init_called = False
        self._finished = False
        self._run_id: str | None = None
        self._run_url: str | None = None
        self._artifact_path: Path | None = None
        self._artifact_result: Any | None = None

    @property
    def enabled(self) -> bool:
        """Whether this adapter is configured to create a W&B run."""

        return self.config.mode != "disabled"

    @property
    def initialized(self) -> bool:
        """Whether ``init`` has been called, including disabled mode."""

        return self._init_called

    @property
    def run_id(self) -> str | None:
        """W&B run id, available after successful enabled initialization."""

        return self._run_id

    @property
    def run_url(self) -> str | None:
        """W&B run URL, if the selected W&B backend exposes one."""

        return self._run_url

    def init(
        self,
        *,
        resolved_config: Mapping[str, Any],
        run_dir: str | Path | None = None,
    ) -> MotionGenWandbLogger:
        """Initialize the configured W&B run and expose its id and URL."""

        if self._finished:
            raise RuntimeError("W&B logger has already finished")
        if self._init_called:
            raise RuntimeError("W&B logger init may only be called once")
        self._init_called = True
        self._validate_mode()
        if not self.enabled:
            return self
        if not isinstance(resolved_config, Mapping):
            raise TypeError("resolved_config must be a mapping")

        wandb_dir = self._resolve_wandb_dir(run_dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._wandb = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            if exc.name != "wandb":
                raise
            raise RuntimeError("W&B logging is enabled but the 'wandb' package is not installed") from exc

        kwargs: dict[str, Any] = {
            "project": self.config.project,
            "config": dict(resolved_config),
            "dir": str(wandb_dir),
            "mode": self.config.mode,
            "job_type": self.config.job_type,
            "save_code": self.config.save_code,
        }
        optional = {
            "entity": self.config.entity,
            "name": self.config.name,
            "group": self.config.group,
            "notes": self.config.notes,
            "id": self.config.run_id,
            "resume": self.config.resume,
        }
        kwargs.update({key: value for key, value in optional.items() if value is not None})
        if self.config.tags:
            kwargs["tags"] = list(self.config.tags)

        self._run = self._wandb.init(**kwargs)
        if self._run is None:
            raise RuntimeError("wandb.init returned no run")
        raw_id = getattr(self._run, "id", None)
        self._run_id = str(raw_id) if raw_id is not None else None
        self._run_url = self._extract_run_url(self._run)
        return self

    def log(self, metrics: Mapping[str, Any], *, step: int) -> dict[str, Any]:
        """Flatten and log one set of scalar-like metrics at ``step``."""

        if not self.enabled:
            return {}
        run = self._require_active_run()
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        flattened = flatten_metrics(metrics)
        if flattened:
            run.log(flattened, step=step)
        return flattened

    def summary(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Flatten metrics and update the W&B run summary."""

        if not self.enabled:
            return {}
        run = self._require_active_run()
        flattened = flatten_metrics(metrics)
        if flattened:
            run.summary.update(flattened)
        return flattened

    def artifact(
        self,
        final_checkpoint: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        aliases: Sequence[str] = ("final",),
    ) -> Any | None:
        """Optionally log exactly one final checkpoint as a model artifact.

        Repeating the call with the same path is idempotent.  Attempting to
        upload a different second checkpoint raises, preventing an accidental
        multi-checkpoint artifact stream on a disk-constrained workstation.
        """

        if not self.enabled or not self.config.log_final_checkpoint_artifact:
            return None
        run = self._require_active_run()
        path = Path(final_checkpoint).expanduser().resolve()
        if self._artifact_path is not None:
            if path == self._artifact_path:
                return self._artifact_result
            raise RuntimeError(
                f"only one final checkpoint artifact may be logged per W&B run; already logged {self._artifact_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"final checkpoint does not exist: {path}")

        artifact_metadata = dict(metadata or {})
        artifact_metadata.update(
            {
                "checkpoint_filename": path.name,
                "checkpoint_sha256": _file_sha256(path),
                "checkpoint_size_bytes": path.stat().st_size,
            }
        )
        artifact_name = self.config.artifact_name or (
            f"{self.config.name or self._run_id or 'motion-generator'}-final-checkpoint"
        )
        artifact_name = _sanitize_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata=artifact_metadata,
        )
        artifact.add_file(str(path), name=path.name)
        self._artifact_result = run.log_artifact(artifact, aliases=list(aliases))
        self._artifact_path = path
        return self._artifact_result

    def finish(self, *, exit_code: int | None = None) -> None:
        """Finish the enabled run; safe to call from ``finally`` more than once."""

        if self._finished:
            return
        self._finished = True
        if self._run is not None:
            self._run.finish(exit_code=exit_code)

    def _validate_mode(self) -> None:
        if self.config.mode not in {"disabled", "online", "offline"}:
            raise ValueError(f"W&B mode must be one of 'disabled', 'online', or 'offline'; got {self.config.mode!r}")

    def _resolve_wandb_dir(self, run_dir: str | Path | None) -> Path:
        if self.config.directory is not None:
            return Path(self.config.directory).expanduser()
        if run_dir is None:
            raise ValueError("run_dir is required when W&B directory is not configured")
        return Path(run_dir).expanduser() / "wandb"

    def _require_active_run(self) -> Any:
        if not self._init_called:
            raise RuntimeError("W&B logger must be initialized before use")
        if self._finished:
            raise RuntimeError("W&B logger has already finished")
        if self._run is None:
            raise RuntimeError("enabled W&B logger has no active run")
        return self._run

    @staticmethod
    def _extract_run_url(run: Any) -> str | None:
        raw_url = getattr(run, "url", None)
        if raw_url:
            return str(raw_url)
        get_url = getattr(run, "get_url", None)
        if callable(get_url):
            fallback = get_url()
            return str(fallback) if fallback else None
        return None


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-.")
    return sanitized or "motion-generator-final-checkpoint"
