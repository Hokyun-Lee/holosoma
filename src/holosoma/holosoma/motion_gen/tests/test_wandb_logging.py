from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from holosoma.motion_gen.wandb_logging import (
    MotionGenWandbLogger,
    WandbLoggingConfig,
    flatten_metrics,
)


class FakeArtifact:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.type = kwargs["type"]
        self.metadata = kwargs["metadata"]
        self.files = []

    def add_file(self, path, *, name):
        self.files.append((path, name))


class FakeRun:
    def __init__(self, *, run_id="run-123", url="https://wandb.invalid/run-123"):
        self.id = run_id
        self.url = url
        self.logged = []
        self.summary = {}
        self.artifacts = []
        self.finish_calls = []

    def log(self, values, *, step):
        self.logged.append((values, step))

    def log_artifact(self, artifact, *, aliases):
        result = SimpleNamespace(artifact=artifact, aliases=aliases)
        self.artifacts.append(result)
        return result

    def finish(self, *, exit_code=None):
        self.finish_calls.append(exit_code)


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self, run=None):
        self.run = run or FakeRun()
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def install_fake_wandb(monkeypatch, fake):
    original_import_module = importlib.import_module

    def import_module(name, package=None):
        if name == "wandb":
            return fake
        return original_import_module(name, package)

    monkeypatch.setattr("holosoma.motion_gen.wandb_logging.importlib.import_module", import_module)


def test_disabled_logger_is_import_free_and_has_no_side_effects(monkeypatch, tmp_path: Path):
    def unexpected_import(name, package=None):
        raise AssertionError(f"disabled logger imported {name}")

    monkeypatch.setattr("holosoma.motion_gen.wandb_logging.importlib.import_module", unexpected_import)
    run_dir = tmp_path / "must_not_be_created"
    logger = MotionGenWandbLogger()

    assert not logger.enabled
    assert logger.init(resolved_config={"seed": 42}, run_dir=run_dir) is logger
    assert logger.initialized
    assert logger.run_id is None
    assert logger.run_url is None
    assert logger.log({"train": {"loss": 1.0}}, step=1) == {}
    assert logger.summary({"best": {"loss": 1.0}}) == {}
    assert logger.artifact(tmp_path / "missing.pt") is None
    logger.finish(exit_code=0)
    logger.finish(exit_code=1)
    assert not run_dir.exists()


def test_online_init_log_summary_and_finish(monkeypatch, tmp_path: Path):
    fake = FakeWandb()
    install_fake_wandb(monkeypatch, fake)
    config = WandbLoggingConfig(
        mode="online",
        entity="hkleetony-dyros",
        project="HoloSomaMotionGenerator",
        name="terrain-feasible-s42",
        group="terrain_feasibility_retrain",
        notes="full-scratch feasibility objective",
        run_id="fixed-id",
        resume="allow",
        tags=["generator", "scratch"],
    )
    logger = MotionGenWandbLogger(config)
    resolved = {"seed": 42, "loss": {"joint_limit": 10.0}}

    logger.init(resolved_config=resolved, run_dir=tmp_path / "run")

    assert logger.enabled
    assert logger.run_id == "run-123"
    assert logger.run_url == "https://wandb.invalid/run-123"
    assert len(fake.init_calls) == 1
    kwargs = fake.init_calls[0]
    assert kwargs["mode"] == "online"
    assert kwargs["entity"] == "hkleetony-dyros"
    assert kwargs["project"] == "HoloSomaMotionGenerator"
    assert kwargs["name"] == "terrain-feasible-s42"
    assert kwargs["group"] == "terrain_feasibility_retrain"
    assert kwargs["notes"] == "full-scratch feasibility objective"
    assert kwargs["id"] == "fixed-id"
    assert kwargs["resume"] == "allow"
    assert kwargs["tags"] == ["generator", "scratch"]
    assert kwargs["config"] == resolved
    assert kwargs["dir"] == str(tmp_path / "run" / "wandb")
    assert (tmp_path / "run" / "wandb").is_dir()

    logged = logger.log(
        {"train": {"loss": {"total": 0.25}, "lr": 1.0e-4}},
        step=7,
    )
    assert logged == {"train/loss/total": 0.25, "train/lr": 1.0e-4}
    assert fake.run.logged == [(logged, 7)]

    summary = logger.summary({"best": {"terrain/max_depth_m": 0.01}})
    assert summary == {"best/terrain/max_depth_m": 0.01}
    assert fake.run.summary == summary

    logger.finish(exit_code=0)
    logger.finish(exit_code=3)
    assert fake.run.finish_calls == [0]
    assert logger.run_id == "run-123"
    assert logger.run_url == "https://wandb.invalid/run-123"


def test_offline_mode_and_explicit_directory(monkeypatch, tmp_path: Path):
    fake = FakeWandb(run=FakeRun(run_id="offline-id", url=""))
    fake.run.get_url = lambda: "wandb-offline://offline-id"
    install_fake_wandb(monkeypatch, fake)
    directory = tmp_path / "explicit_wandb"
    logger = MotionGenWandbLogger(WandbLoggingConfig(mode="offline", directory=str(directory), save_code=False))

    logger.init(resolved_config={}, run_dir=None)

    assert fake.init_calls[0]["mode"] == "offline"
    assert fake.init_calls[0]["dir"] == str(directory)
    assert not fake.init_calls[0]["save_code"]
    assert logger.run_id == "offline-id"
    assert logger.run_url == "wandb-offline://offline-id"


def test_final_checkpoint_artifact_is_hashed_and_single(monkeypatch, tmp_path: Path):
    fake = FakeWandb()
    install_fake_wandb(monkeypatch, fake)
    logger = MotionGenWandbLogger(
        WandbLoggingConfig(
            mode="online",
            name="terrain feasible / seed 42",
            log_final_checkpoint_artifact=True,
            artifact_name="selected final @ 200k",
        )
    )
    logger.init(resolved_config={}, run_dir=tmp_path / "run")
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"final-checkpoint")

    result = logger.artifact(
        checkpoint,
        metadata={"step": 200_000, "checkpoint_sha256": "must-be-replaced"},
    )

    assert result is fake.run.artifacts[0]
    artifact = result.artifact
    assert artifact.name == "selected-final-200k"
    assert artifact.type == "model"
    assert artifact.metadata["step"] == 200_000
    assert artifact.metadata["checkpoint_filename"] == "final.pt"
    assert artifact.metadata["checkpoint_size_bytes"] == len(b"final-checkpoint")
    assert artifact.metadata["checkpoint_sha256"] == hashlib.sha256(b"final-checkpoint").hexdigest()
    assert artifact.files == [(str(checkpoint.resolve()), "final.pt")]
    assert result.aliases == ["final"]
    assert logger.artifact(checkpoint) is result
    assert len(fake.run.artifacts) == 1

    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="only one final checkpoint"):
        logger.artifact(other)


def test_artifact_option_disabled_does_not_touch_path(monkeypatch, tmp_path: Path):
    fake = FakeWandb()
    install_fake_wandb(monkeypatch, fake)
    logger = MotionGenWandbLogger(WandbLoggingConfig(mode="online"))
    logger.init(resolved_config={}, run_dir=tmp_path / "run")

    assert logger.artifact(tmp_path / "does-not-exist.pt") is None
    assert not fake.run.artifacts


def test_invalid_usage_and_missing_dependency(monkeypatch, tmp_path: Path):
    invalid = MotionGenWandbLogger(WandbLoggingConfig(mode="invalid"))
    with pytest.raises(ValueError, match="mode must be one of"):
        invalid.init(resolved_config={}, run_dir=tmp_path)

    def missing_wandb(name, package=None):
        error = ModuleNotFoundError("No module named 'wandb'")
        error.name = "wandb"
        raise error

    monkeypatch.setattr("holosoma.motion_gen.wandb_logging.importlib.import_module", missing_wandb)
    enabled = MotionGenWandbLogger(WandbLoggingConfig(mode="online"))
    with pytest.raises(RuntimeError, match="not installed"):
        enabled.init(resolved_config={}, run_dir=tmp_path / "missing")


def test_active_run_guards_and_step_validation(monkeypatch, tmp_path: Path):
    fake = FakeWandb()
    install_fake_wandb(monkeypatch, fake)
    logger = MotionGenWandbLogger(WandbLoggingConfig(mode="online"))

    with pytest.raises(RuntimeError, match="initialized"):
        logger.log({"loss": 1.0}, step=0)
    logger.init(resolved_config={}, run_dir=tmp_path / "run")
    with pytest.raises(RuntimeError, match="only be called once"):
        logger.init(resolved_config={}, run_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="non-negative integer"):
        logger.log({"loss": 1.0}, step=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        logger.log({"loss": 1.0}, step=True)
    logger.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        logger.summary({"loss": 1.0})

    finished_before_init = MotionGenWandbLogger(WandbLoggingConfig(mode="online"))
    finished_before_init.finish()
    with pytest.raises(RuntimeError, match="already finished"):
        finished_before_init.init(resolved_config={}, run_dir=tmp_path / "too_late")


def test_flatten_metrics_handles_prefix_and_rejects_collisions():
    assert flatten_metrics({"loss": {"total": 1.0}}, prefix="val") == {"val/loss/total": 1.0}
    assert flatten_metrics({"empty": {}, "value": None}) == {"value": None}
    with pytest.raises(TypeError, match="must be a mapping"):
        flatten_metrics([("loss", 1.0)])
    with pytest.raises(ValueError, match="separator"):
        flatten_metrics({"loss": 1.0}, separator="")
    with pytest.raises(ValueError, match="collision"):
        flatten_metrics({"a/b": 1.0, "a": {"b": 2.0}})
