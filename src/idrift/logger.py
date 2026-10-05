"""RunLogger & BatchSummary (PRD §6, FR-3). One log per run, atomic writes."""
from __future__ import annotations

import datetime as dt
import json
import os
import secrets
from pathlib import Path
from typing import Any


def generate_run_id(batch_name: str) -> str:
    """Format: {batch_name}_{YYYYmmddTHHMMSS}_{4 hex}."""
    now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    rand_hex = secrets.token_hex(2)  # 4 chars hex
    return f"{batch_name}_{now_str}_{rand_hex}"


_SENSITIVE_SUBSTRINGS = ("api_key", "apikey", "api-key", "secret", "token", "password", "passwd", "auth", "bearer")


def _sanitize_dict(obj: Any) -> Any:
    """Pastikan tak ada key sensitif (key/token/secret/...) di dump snapshot."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            kl = k.lower()
            if kl == "api_key_env":
                cleaned[k] = _sanitize_dict(v)
                continue
            if any(s in kl for s in _SENSITIVE_SUBSTRINGS):
                continue
            cleaned[k] = _sanitize_dict(v)
        return cleaned
    if isinstance(obj, list):
        return [_sanitize_dict(i) for i in obj]
    return obj


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Tulis atomik via .tmp lalu os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    """Tulis text atomik via .tmp lalu os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


class RunLogger:
    """Kelola state run, partial checkpoints, dan log final."""

    def __init__(self, log_dir: str | Path, batch_name: str, run_id: str | None = None) -> None:
        self.batch_name = batch_name
        self.run_id = run_id or generate_run_id(batch_name)
        self.batch_dir = Path(log_dir) / batch_name
        self.final_json_path = self.batch_dir / f"{self.run_id}.json"
        self.partial_json_path = self.batch_dir / f"{self.run_id}.partial.json"
        self.transcript_path = self.batch_dir / f"{self.run_id}.transcript.md"

        self._started_monotonic: float | None = None
        self.data: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "batch_name": self.batch_name,
            "status": "initialized",
            "started_at": "",
            "finished_at": "",
            "duration_sec": 0,
            "condition": {},
            "config_snapshot": {},
            "models": {},
            "prompts": {},
            "conversation": [],
            "snapshots": [],
            "errors": [],
            "totals": {"api_calls": 0, "tokens_in": 0, "tokens_out": 0},
        }

    def start(
        self,
        config_snapshot: dict[str, Any],
        models_info: dict[str, Any],
        prompts: dict[str, Any],
        condition: dict[str, Any] | None = None,
    ) -> None:
        """Mulai run dan set status running."""
        now = dt.datetime.now(dt.timezone.utc)
        self._started_monotonic = dt.datetime.now().timestamp()
        self.data["status"] = "running"
        self.data["started_at"] = now.isoformat()
        self.data["config_snapshot"] = _sanitize_dict(config_snapshot)
        self.data["models"] = _sanitize_dict(models_info)
        self.data["prompts"] = prompts
        if condition is not None:
            self.data["condition"] = condition
        else:
            self.data["condition"] = {
                "research_question": config_snapshot.get("research_question", "RQ1"),
                "persona_condition": config_snapshot.get("persona_condition", "none"),
                "size_group": models_info.get("agent_a", {}).get("size_group"),
                "family": models_info.get("agent_a", {}).get("family"),
            }
        self._save_partial()

    def add_theme_record(self, record: dict[str, Any]) -> None:
        """Catat hasil satu tema dan simpan checkpoint partial."""
        self.data["conversation"].append(record)
        self._recompute_totals()
        self._save_partial()

    def add_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Catat satu snapshot kuesioner dan simpan checkpoint partial."""
        self.data["snapshots"].append(snapshot)
        self._recompute_totals()
        self._save_partial()

    def add_error(self, stage: str, message: str, retries: int = 0) -> None:
        """Catat error."""
        self.data["errors"].append({"stage": stage, "message": str(message), "retries": retries})
        self._save_partial()

    def _recompute_totals(self) -> None:
        """Hitung api_calls, tokens_in, tokens_out dari percakapan & snapshot."""
        api_calls = 0
        t_in = 0
        t_out = 0

        for conv in self.data["conversation"]:
            for side in ("agent_a", "agent_b"):
                ag = conv.get(side)
                if ag and isinstance(ag, dict):
                    api_calls += 1
                    u = ag.get("usage", {})
                    t_in += int(u.get("in", 0) or 0)
                    t_out += int(u.get("out", 0) or 0)

        for sn in self.data["snapshots"]:
            for q in sn.get("questionnaires", []):
                for r in q.get("repeats", []):
                    api_calls += 1
                    u = r.get("usage", {})
                    t_in += int(u.get("in", 0) or 0)
                    t_out += int(u.get("out", 0) or 0)

        self.data["totals"] = {"api_calls": api_calls, "tokens_in": t_in, "tokens_out": t_out}

    def _render_transcript(self) -> str:
        """Bangun isi file .transcript.md."""
        lines = [
            f"# Transcript — {self.run_id}",
            f"- Batch: `{self.batch_name}`",
            f"- Status: `{self.data.get('status', '')}`",
            f"- Started: `{self.data.get('started_at', '')}`",
            f"- Finished: `{self.data.get('finished_at', '')}`",
            "",
            "---",
            "",
        ]
        for c in self.data.get("conversation", []):
            tid = c.get("theme_id")
            ttext = c.get("theme_text")
            lines.append(f"## Question {tid}: {ttext}\n")
            a_text = c.get("agent_a", {}).get("text", "")
            b_text = c.get("agent_b", {}).get("text", "")
            lines.append(f"**Agent A:**\n{a_text}\n")
            lines.append(f"**Agent B:**\n{b_text}\n")
            lines.append("---\n")
        return "\n".join(lines)

    def _save_partial(self) -> None:
        """Simpan .partial.json atomik."""
        _atomic_write_json(self.partial_json_path, self.data)

    def finish(self, status: str = "completed") -> None:
        """Selesai run sukses: tulis .json + .transcript.md, hapus .partial.json."""
        now = dt.datetime.now(dt.timezone.utc)
        self.data["status"] = status
        self.data["finished_at"] = now.isoformat()
        if self._started_monotonic is not None:
            self.data["duration_sec"] = max(0, int(dt.datetime.now().timestamp() - self._started_monotonic))
        self._recompute_totals()

        # Tulis JSON final atomik
        _atomic_write_json(self.final_json_path, self.data)

        # Tulis Transcript Markdown
        _atomic_write_text(self.transcript_path, self._render_transcript())

        # Hapus partial jika ada
        if self.partial_json_path.exists():
            try:
                self.partial_json_path.unlink()
            except OSError:
                pass

    def fail(self, exc: BaseException | None = None, status: str = "failed") -> None:
        """Gagal atau di-interrupt: partial dipertahankan, transcript parsial ditulis."""
        now = dt.datetime.now(dt.timezone.utc)
        self.data["status"] = status
        self.data["finished_at"] = now.isoformat()
        if self._started_monotonic is not None:
            self.data["duration_sec"] = max(0, int(dt.datetime.now().timestamp() - self._started_monotonic))
        if exc is not None:
            msg = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            self.add_error(stage="runtime", message=msg)
        self._recompute_totals()

        # Update partial
        self._save_partial()

        # Tulis transcript parsial
        _atomic_write_text(self.transcript_path, self._render_transcript())


class BatchSummary:
    """Ringkasan semua run di logs/{batch_name}/_batch_summary.json."""

    def __init__(self, log_dir: str | Path, batch_name: str) -> None:
        self.batch_name = batch_name
        self.batch_dir = Path(log_dir) / batch_name
        self.summary_path = self.batch_dir / "_batch_summary.json"

    def generate(self) -> dict[str, Any]:
        """Kumpulkan semua file log run dan buat ringkasan."""
        runs: list[dict[str, Any]] = []
        if self.batch_dir.exists():
            for p in sorted(self.batch_dir.glob("*.json")):
                if p.name.startswith("_") or p.name.endswith(".partial.json"):
                    continue
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    runs.append({
                        "run_id": data.get("run_id", p.stem),
                        "status": data.get("status", "unknown"),
                        "duration_sec": data.get("duration_sec", 0),
                        "errors_count": len(data.get("errors", [])),
                        "totals": data.get("totals", {}),
                    })
                except Exception:
                    continue

        summary = {
            "batch_name": self.batch_name,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "total_runs": len(runs),
            "completed_runs": sum(1 for r in runs if r["status"] == "completed"),
            "runs": runs,
        }
        _atomic_write_json(self.summary_path, summary)
        return summary
