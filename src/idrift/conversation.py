"""Conversation builder + loop (PRD §3.3). Turn order A dulu, lalu B."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

Agent = Literal["A", "B"]


def _theme_text(themes: Sequence[Any], idx: int) -> str:
    """Ambil teks tema. Dukung str atau dict {text}."""
    t = themes[idx]
    if isinstance(t, dict):
        return str(t.get("text", ""))
    return str(t)


def _agent_text(entry: dict[str, Any], agent: Agent) -> str | None:
    """Ambil jawaban agent dari history entry. None jika key tak ada."""
    key = f"agent_{agent.lower()}"
    alt = agent.upper()
    found = False
    v: Any = None
    if isinstance(entry, dict):
        for k in (key, agent.lower(), alt):
            if k in entry:
                v = entry[k]
                found = True
                break
    if not found:
        return None
    if isinstance(v, dict):
        return str(v.get("text", ""))
    return str(v) if v is not None else ""


def _q(themes: Sequence[Any], idx: int) -> str:
    """Format tema: 'Question {t} : {text}'. t 1-based."""
    return f"Question {idx + 1} : {_theme_text(themes, idx)}"


def build_messages(
    agent: Agent,
    theme_idx: int,
    history: list[dict[str, Any]],
    themes: Sequence[Any],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Bangun pesan API untuk satu agent satu tema.

    history: record selesai. Untuk agent B, history[theme_idx]
      wajib ada agent_a (jawaban A tema sama, karena B jalan kedua).
    """
    if agent not in ("A", "B"):
        raise ValueError("agent harus 'A' atau 'B'")
    if not 0 <= theme_idx < len(themes):
        raise ValueError("theme_idx di luar themes")
    if agent == "A" and len(history) < theme_idx:
        raise ValueError("history kurang untuk agent A")
    if agent == "B":
        if len(history) < theme_idx + 1:
            raise ValueError("history B butuh agent_a tema sama")
        if _agent_text(history[theme_idx], "A") is None:
            raise ValueError("history B butuh agent_a tema sama")

    raw: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if agent == "A":
        for k in range(theme_idx):
            a = _agent_text(history[k], "A")
            b = _agent_text(history[k], "B")
            if a is None or b is None:
                raise ValueError(f"history[{k}] butuh agent_a + agent_b")
            raw.append({"role": "user", "content": _q(themes, k)})
            raw.append({"role": "assistant", "content": a})
            raw.append({"role": "user", "content": b})
        raw.append({"role": "user", "content": _q(themes, theme_idx)})
    else:
        for k in range(theme_idx):
            a = _agent_text(history[k], "A")
            b = _agent_text(history[k], "B")
            if a is None or b is None:
                raise ValueError(f"history[{k}] butuh agent_a + agent_b")
            raw.append({"role": "user", "content": _q(themes, k)})
            raw.append({"role": "user", "content": a})
            raw.append({"role": "assistant", "content": b})
        raw.append({"role": "user", "content": _q(themes, theme_idx)})
        a_curr = _agent_text(history[theme_idx], "A")
        raw.append({"role": "user", "content": a_curr if a_curr is not None else ""})

    merged: list[dict[str, str]] = [raw[0]]
    for m in raw[1:]:
        prev = merged[-1]
        if m["role"] == "user" and prev["role"] == "user":
            prev["content"] += "\n\n" + m["content"]
        else:
            merged.append({"role": m["role"], "content": m["content"]})
    return merged


def _record_side(res: Any) -> dict[str, Any]:
    """Normalisasi hasil chat ke record PRD §FR-1."""
    if isinstance(res, str):
        return {"text": res, "usage": {"in": 0, "out": 0}, "latency_ms": 0, "finish_reason": "stop"}
    text = str(getattr(res, "text", ""))
    u_in = int(getattr(res, "usage_in", 0) or 0)
    u_out = int(getattr(res, "usage_out", 0) or 0)
    if isinstance(res, dict):
        text = str(res.get("text", text))
        usage = res.get("usage", {})
        u_in = int(usage.get("in", u_in))
        u_out = int(usage.get("out", u_out))
    return {
        "text": text,
        "usage": {"in": u_in, "out": u_out},
        "latency_ms": int(getattr(res, "latency_ms", 0) or 0) if not isinstance(res, dict) else int(res.get("latency_ms", 0)),
        "finish_reason": str(getattr(res, "finish_reason", "stop") or "stop") if not isinstance(res, dict) else str(res.get("finish_reason", "stop")),
    }


def run_conversation(
    client_a: Any,
    client_b: Any,
    themes: Sequence[Any],
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    on_theme_done: Callable[[int, dict[str, Any]], None] | None = None,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Loop 36 tema. A jawab dulu tiap tema, B lihat jawaban A. Dukung resume."""
    records: list[dict[str, Any]] = list(existing_records or [])
    start_idx = len(records)
    for idx in range(start_idx, len(themes)):
        theme_id = idx + 1
        theme_text = _theme_text(themes, idx)
        msgs_a = build_messages("A", idx, records, themes, system_prompt)
        res_a = client_a.chat(msgs_a, temperature=temperature, max_tokens=max_tokens)
        text_a = _record_side(res_a)["text"]
        ext = [*records, {"agent_a": text_a, "agent_b": ""}]
        msgs_b = build_messages("B", idx, ext, themes, system_prompt)
        res_b = client_b.chat(msgs_b, temperature=temperature, max_tokens=max_tokens)
        rec = {
            "theme_id": theme_id,
            "theme_text": theme_text,
            "agent_a": _record_side(res_a),
            "agent_b": _record_side(res_b),
        }
        records.append(rec)
        if on_theme_done is not None:
            on_theme_done(theme_id, rec)
    return records
