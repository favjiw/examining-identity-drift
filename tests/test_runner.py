from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
from typer.testing import CliRunner

from idrift.cli import app
from idrift.config import load_config
from idrift.runner import estimate_calls, run_batch


class MockClient:
    def __init__(self, name: str = "agent"):
        self.name = name
        self.calls = 0

    def chat(self, messages, temperature=0.7, max_tokens=512):
        self.calls += 1
        return {
            "text": f"{self.name}_ans_{self.calls}",
            "usage": {"in": 12, "out": 6},
            "latency_ms": 15,
            "finish_reason": "stop",
        }


def test_estimate_numbers():
    cfg = load_config("configs/mock.yaml")
    # In mock.yaml: n_runs=2, include=all (15), repeats=10, snapshots=[12,24,36] (3), measure=[A,B] (2)
    # conv = 2 * 36 * 2 = 144
    # quest = 2 * 3 * 15 * 10 * 2 = 1800
    # total = 1944
    est = estimate_calls(cfg)
    assert est["conversation"] == 144
    assert est["questionnaire"] == 1800
    assert est["total"] == 1944
    assert est["n_runs"] == 2
    assert est["n_questionnaires"] == 15


def test_run_batch_quick_creates_all_files_without_api_key(tmp_path):
    cfg = load_config("configs/mock.yaml")
    cfg.runtime.log_dir = str(tmp_path)
    
    clients = []

    def mock_factory(agent_cfg):
        c = MockClient(name=agent_cfg.model)
        clients.append(c)
        return c

    # Jalankan quick (1 run)
    run_ids = run_batch(cfg, quick=True, client_factory=mock_factory)
    assert len(run_ids) == 1
    rid = run_ids[0]

    batch_dir = tmp_path / cfg.batch_name
    final_json = batch_dir / f"{rid}.json"
    transcript_md = batch_dir / f"{rid}.transcript.md"
    summary_json = batch_dir / "_batch_summary.json"
    partial_json = batch_dir / f"{rid}.partial.json"

    assert final_json.exists(), "JSON final wajib dibuat otomatis"
    assert transcript_md.exists(), "Transcript wajib dibuat otomatis"
    assert summary_json.exists(), "Batch summary wajib dibuat otomatis"
    assert not partial_json.exists(), "Partial json harus dihapus saat selesai"

    with open(final_json, encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "completed"
    assert len(data["conversation"]) == 36
    assert data["totals"]["api_calls"] == 72


def test_resume_does_not_re_call_completed_themes(tmp_path):
    cfg = load_config("configs/mock.yaml")
    cfg.n_runs = 1
    cfg.runtime.log_dir = str(tmp_path)
    batch_dir = tmp_path / cfg.batch_name
    batch_dir.mkdir(parents=True)

    # 1. Bikin file .partial.json tiruan dengan 10 tema selesai
    run_id = f"{cfg.batch_name}_20261003T120000_1234"
    partial_path = batch_dir / f"{run_id}.partial.json"
    
    fake_conv = []
    for i in range(1, 11):
        fake_conv.append({
            "theme_id": i,
            "theme_text": f"Theme {i}",
            "agent_a": {"text": f"A_{i}", "usage": {"in": 1, "out": 1}, "latency_ms": 1, "finish_reason": "stop"},
            "agent_b": {"text": f"B_{i}", "usage": {"in": 1, "out": 1}, "latency_ms": 1, "finish_reason": "stop"},
        })

    partial_data = {
        "schema_version": "1.0",
        "run_id": run_id,
        "batch_name": cfg.batch_name,
        "status": "incomplete",
        "started_at": "2026-10-03T12:00:00+00:00",
        "finished_at": "",
        "duration_sec": 0,
        "condition": {},
        "config_snapshot": {},
        "models": {},
        "prompts": {},
        "conversation": fake_conv,
        "snapshots": [],
        "errors": [],
        "totals": {"api_calls": 20, "tokens_in": 20, "tokens_out": 20},
    }
    partial_path.write_text(json.dumps(partial_data), encoding="utf-8")

    # 2. Jalankan resume
    client_a = MockClient(name="client_a")
    client_b = MockClient(name="client_b")
    toggle = [True]

    def mock_factory(agent_cfg):
        # alternate call: first for A, second for B
        res = client_a if toggle[0] else client_b
        toggle[0] = not toggle[0]
        return res

    run_ids = run_batch(cfg, resume_dir=batch_dir, client_factory=mock_factory)
    assert run_id in run_ids

    # Final JSON sekarang harus ada 36 tema
    final_json = batch_dir / f"{run_id}.json"
    assert final_json.exists()
    assert not partial_path.exists()

    with open(final_json, encoding="utf-8") as f:
        finished = json.load(f)

    assert len(finished["conversation"]) == 36
    # Tema 1..10 tidak dipanggil ulang -> hanya tema 11..36 (26 tema)
    assert client_a.calls == 26
    assert client_b.calls == 26


def test_cli_estimate():
    runner = CliRunner()
    result = runner.invoke(app, ["estimate", "--config", "configs/mock.yaml"])
    assert result.exit_code == 0
    assert "Estimasi panggilan API" in result.stdout
    assert "1944" in result.stdout
    assert "144" in result.stdout
    assert "1800" in result.stdout
