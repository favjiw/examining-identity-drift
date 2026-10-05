from __future__ import annotations

from typing import Any
import pytest

from idrift.conversation import build_messages, run_conversation


class MockLLMClient:
    """Mock client rekam prompt dan return respons deterministik."""

    def __init__(self, prefix: str = "ans"):
        self.prefix = prefix
        self.calls: list[dict[str, Any]] = []
        self.counter = 0

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 512):
        self.counter += 1
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        ans = f"{self.prefix}_{self.counter}"
        return {
            "text": ans,
            "usage": {"in": 10, "out": 5},
            "latency_ms": 12,
            "finish_reason": "stop",
        }


THEMES = [
    "Given the choice of anyone in the world, whom would you want as a dinner guest?",
    "Would you like to be famous? In what way?",
    "Before making a telephone call, do you ever rehearse what you are going to say?",
]
SYS_PROMPT = "You are now sharing your thoughts on the question with your partner.\nYou only reply briefly to your thoughts only for a given question."


def test_theme_1_agent_a():
    """Agent A, tema 1: system, lalu user Q1."""
    msgs = build_messages("A", 0, [], THEMES, SYS_PROMPT)
    assert msgs == [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question 1 : {THEMES[0]}"},
    ]


def test_theme_1_agent_b_verbatim():
    """Agent B, tema 1: user digabung Q1 + jawaban A."""
    history = [{"agent_a": "Jawaban A1", "agent_b": ""}]
    msgs = build_messages("B", 0, history, THEMES, SYS_PROMPT)
    expected = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question 1 : {THEMES[0]}\n\nJawaban A1"},
    ]
    assert msgs == expected


def test_theme_2_agent_a_verbatim():
    """Agent A, tema 2: system, user Q1, assistant A1, user (B1 + Q2)."""
    history = [{"agent_a": "Jawaban A1", "agent_b": "Jawaban B1"}]
    msgs = build_messages("A", 1, history, THEMES, SYS_PROMPT)
    expected = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question 1 : {THEMES[0]}"},
        {"role": "assistant", "content": "Jawaban A1"},
        {"role": "user", "content": f"Jawaban B1\n\nQuestion 2 : {THEMES[1]}"},
    ]
    assert msgs == expected


def test_theme_2_agent_b_verbatim():
    """Agent B, tema 2: system, user (Q1 + A1), assistant B1, user (Q2 + A2)."""
    history = [
        {"agent_a": "Jawaban A1", "agent_b": "Jawaban B1"},
        {"agent_a": "Jawaban A2", "agent_b": ""},
    ]
    msgs = build_messages("B", 1, history, THEMES, SYS_PROMPT)
    expected = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question 1 : {THEMES[0]}\n\nJawaban A1"},
        {"role": "assistant", "content": "Jawaban B1"},
        {"role": "user", "content": f"Question 2 : {THEMES[1]}\n\nJawaban A2"},
    ]
    assert msgs == expected


def test_no_consecutive_same_roles():
    """Tak boleh ada role berurutan sama selain system di awal."""
    history = [
        {"agent_a": f"A{i}", "agent_b": f"B{i}"} for i in range(1, 3)
    ]
    # Uji Agent A tema 3
    msgs_a = build_messages("A", 2, history, THEMES, SYS_PROMPT)
    for i in range(1, len(msgs_a) - 1):
        assert msgs_a[i]["role"] != msgs_a[i + 1]["role"], f"Role sama berurutan di idx {i}"

    # Uji Agent B tema 3
    hist_b = [*history, {"agent_a": "A3", "agent_b": ""}]
    msgs_b = build_messages("B", 2, hist_b, THEMES, SYS_PROMPT)
    for i in range(1, len(msgs_b) - 1):
        assert msgs_b[i]["role"] != msgs_b[i + 1]["role"], f"Role sama berurutan di idx {i}"


def test_perspective_separated():
    """Ucapan sendiri harus assistant, ucapan partner user."""
    hist_a = [{"agent_a": "AKU_A", "agent_b": "DIA_B"}]
    msgs_a = build_messages("A", 1, hist_a, THEMES, SYS_PROMPT)
    assert msgs_a[2] == {"role": "assistant", "content": "AKU_A"}
    assert "DIA_B" in msgs_a[3]["content"]

    hist_b = [{"agent_a": "DIA_A", "agent_b": "AKU_B"}, {"agent_a": "DIA_A2", "agent_b": ""}]
    msgs_b = build_messages("B", 1, hist_b, THEMES, SYS_PROMPT)
    assert msgs_b[2] == {"role": "assistant", "content": "AKU_B"}
    assert "DIA_A" in msgs_b[1]["content"]
    assert "DIA_A2" in msgs_b[3]["content"]


def test_run_conversation_36_themes():
    """36 tema menghasilkan 36 record, callback dipanggil 36x."""
    themes_36 = [f"Theme {i}" for i in range(1, 37)]
    client_a = MockLLMClient(prefix="agentA")
    client_b = MockLLMClient(prefix="agentB")

    cb_calls = []

    def on_done(tid, rec):
        cb_calls.append((tid, rec["theme_id"]))

    records = run_conversation(
        client_a=client_a,
        client_b=client_b,
        themes=themes_36,
        system_prompt=SYS_PROMPT,
        temperature=0.7,
        max_tokens=256,
        on_theme_done=on_done,
    )

    assert len(records) == 36
    assert len(cb_calls) == 36
    assert cb_calls[0] == (1, 1)
    assert cb_calls[-1] == (36, 36)

    # 36 tema x 2 agent = 72 panggilan
    assert len(client_a.calls) == 36
    assert len(client_b.calls) == 36

    # Cek record pertama & terakhir
    first = records[0]
    assert first["theme_id"] == 1
    assert first["theme_text"] == "Theme 1"
    assert first["agent_a"]["text"] == "agentA_1"
    assert first["agent_b"]["text"] == "agentB_1"
    assert first["agent_a"]["usage"] == {"in": 10, "out": 5}

    last = records[35]
    assert last["theme_id"] == 36
    assert last["agent_a"]["text"] == "agentA_36"
    assert last["agent_b"]["text"] == "agentB_36"
