"""The Stage 3 script must refuse to run without --confirm, without a freeze, or when the lock exists."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage3_holdout.py"
spec = importlib.util.spec_from_file_location("stage3", SCRIPT)
stage3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3)


def test_refuses_without_confirm_or_freeze(tmp_path):
    errs = stage3.check_gates(False, tmp_path)
    assert any("--confirm" in e for e in errs) and any("FREEZE.json" in e for e in errs) and any("Stage 1" in e for e in errs)


def test_refuses_when_lock_exists(tmp_path):
    (tmp_path / "HOLDOUT_LOCK").write_text("{}")
    errs = stage3.check_gates(True, tmp_path)
    assert any("already been run" in e for e in errs)


def test_detects_config_change_after_freeze(tmp_path):
    root = tmp_path / "repo"; root.mkdir()
    (root / "config.yaml").write_text("a: 1\n")
    (tmp_path / "STAGE1_PASSED").write_text("x")
    (tmp_path / "FREEZE.json").write_text('{"config.yaml": "deadbeef"}')
    errs = stage3.check_gates(True, tmp_path, root=root)
    assert any("hash mismatch" in e for e in errs)
