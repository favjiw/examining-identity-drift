from __future__ import annotations

import json
import pytest

from idrift.logger import RunLogger, BatchSummary, generate_run_id


def test_generate_run_id():
    rid = generate_run_id("test_batch")
    assert rid.startswith("test_batch_")
    parts = rid.split("_")
    # batch, timestamp, 4 hex
    assert len(parts) >= 3


def test_run_success_creates_json_and_transcript_removes_partial(tmp_path):
    logger = RunLogger(log_dir=tmp_path, batch_name="batch1")
    logger.start(
        config_snapshot={"research_question": "RQ1", "persona_condition": "none"},
        models_info={"agent_a": {"model": "m1"}, "agent_b": {"model": "m1"}},
        prompts={"conversation_system": "sys"},
    )
    
    # Check partial dibuat saat start
    assert logger.partial_json_path.exists()
    assert not logger.final_json_path.exists()
    assert not logger.transcript_path.exists()

    # Tambah 36 tema
    for i in range(1, 37):
        logger.add_theme_record({
            "theme_id": i,
            "theme_text": f"Theme {i}",
            "agent_a": {"text": f"A_{i}", "usage": {"in": 5, "out": 2}, "latency_ms": 10, "finish_reason": "stop"},
            "agent_b": {"text": f"B_{i}", "usage": {"in": 7, "out": 3}, "latency_ms": 11, "finish_reason": "stop"},
        })

    logger.finish()

    # Final files harus ada
    assert logger.final_json_path.exists()
    assert logger.transcript_path.exists()
    # Partial harus terhapus
    assert not logger.partial_json_path.exists()

    # Cek isi JSON
    with open(logger.final_json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "completed"
    assert len(data["conversation"]) == 36
    assert data["totals"]["api_calls"] == 72
    assert data["totals"]["tokens_in"] == (5 + 7) * 36
    assert data["totals"]["tokens_out"] == (2 + 3) * 36

    # Cek transcript
    transcript_text = logger.transcript_path.read_text(encoding="utf-8")
    assert "## Question 1: Theme 1" in transcript_text
    assert "**Agent A:**\nA_1" in transcript_text
    assert "**Agent B:**\nB_1" in transcript_text
    assert "## Question 36: Theme 36" in transcript_text


def test_failure_leaves_partial_and_no_final_json(tmp_path):
    logger = RunLogger(log_dir=tmp_path, batch_name="batch_fail")
    logger.start(
        config_snapshot={"research_question": "RQ1"},
        models_info={"agent_a": {"model": "m1"}},
        prompts={"conversation_system": "sys"},
    )

    logger.add_theme_record({
        "theme_id": 1,
        "theme_text": "Theme 1",
        "agent_a": {"text": "A1"},
        "agent_b": {"text": "B1"},
    })

    # Simulasikan exception / interrupt di tengah run
    try:
        raise KeyboardInterrupt("Simulated Ctrl+C")
    except KeyboardInterrupt as exc:
        logger.fail(exc=exc, status="incomplete")

    # .final.json TIDAK boleh ada
    assert not logger.final_json_path.exists()
    # .partial.json WAJIB ada
    assert logger.partial_json_path.exists()
    # transcript parsial tetap ada
    assert logger.transcript_path.exists()

    with open(logger.partial_json_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "incomplete"
    assert len(data["conversation"]) == 1
    assert any("KeyboardInterrupt" in err["message"] for err in data["errors"])


def test_no_api_key_leaked_in_any_file(tmp_path):
    secret_key = "sk-secret-1234567890-very-secret"
    logger = RunLogger(log_dir=tmp_path, batch_name="batch_sec")
    
    # Kirim config yang ceroboh menyertakan key
    logger.start(
        config_snapshot={
            "api_key": secret_key,
            "secret_token": "token-xyz",
            "api_key_env": "OPENAI_API_KEY",
        },
        models_info={
            "agent_a": {"model": "gpt-4", "api_key": secret_key, "api_key_env": "OPENAI_API_KEY"}
        },
        prompts={"conversation_system": "sys"},
    )

    logger.add_theme_record({
        "theme_id": 1,
        "theme_text": "Theme 1",
        "agent_a": {"text": "A1"},
        "agent_b": {"text": "B1"},
    })
    logger.finish()

    final_content = logger.final_json_path.read_text(encoding="utf-8")
    transcript_content = logger.transcript_path.read_text(encoding="utf-8")

    assert secret_key not in final_content
    assert "token-xyz" not in final_content
    assert secret_key not in transcript_content
    assert "OPENAI_API_KEY" in final_content


def test_batch_summary_aggregation(tmp_path):
    batch_name = "multi_run_batch"
    
    # Bikin 2 run sukses
    for i in range(2):
        l = RunLogger(log_dir=tmp_path, batch_name=batch_name, run_id=f"run_{i}")
        l.start({}, {}, {})
        l.add_theme_record({
            "theme_id": 1,
            "agent_a": {"text": "a", "usage": {"in": 10, "out": 5}},
            "agent_b": {"text": "b", "usage": {"in": 20, "out": 10}},
        })
        l.finish()

    summary_gen = BatchSummary(log_dir=tmp_path, batch_name=batch_name)
    summary = summary_gen.generate()

    assert summary["batch_name"] == batch_name
    assert summary["total_runs"] == 2
    assert summary["completed_runs"] == 2
    assert len(summary["runs"]) == 2
    assert summary_gen.summary_path.exists()
