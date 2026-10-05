"""Batch runner (PRD FR-1, §3.3, FR-3). Tanpa kuesioner di tahap 4."""
from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from idrift.config import ExperimentConfig
from idrift.conversation import run_conversation
from idrift.llm_client import LLMClient
from idrift.logger import BatchSummary, RunLogger


DEFAULT_SYSTEM_PROMPT = "You are now sharing your thoughts on the question with your partner.\nYou only reply briefly to your thoughts only for a given question."


def estimate_calls(config: ExperimentConfig) -> dict[str, Any]:
    """Estimate API calls (PRD §8)."""
    conv_calls = config.n_runs * 36 * 2
    n_q = 0 if config.questionnaire.include is None else (15 if config.questionnaire.include == "all" else len(config.questionnaire.include))  # type: ignore[arg-type]
    quest_calls = config.n_runs * len(config.questionnaire.snapshots) * n_q * config.questionnaire.repeats * len(config.questionnaire.measure_agents)
    total = conv_calls + quest_calls
    return {
        "conversation": conv_calls,
        "questionnaire": quest_calls,
        "total": total,
        "n_runs": config.n_runs,
        "n_questionnaires": n_q,
        "repeats": config.questionnaire.repeats,
        "measure_agents": len(config.questionnaire.measure_agents),
    }


def _load_themes(themes_file: str | Path) -> list[Any]:
    path = Path(themes_file)
    if not path.exists():
        return [f"Theme {i}" for i in range(1, 37)]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data, list) and isinstance(data[0], dict) and "text" in data[0]:
        return data
    return data


def _load_system_prompt(cfg: ExperimentConfig) -> str:
    path = Path(cfg.conversation.system_prompt_file)
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return DEFAULT_SYSTEM_PROMPT


def _get_models_info(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "agent_a": {"provider": cfg.agent_a.provider, "model": cfg.agent_a.model, "persona": cfg.agent_a.persona, "size_group": cfg.agent_a.size_group, "family": cfg.agent_a.family},
        "agent_b": {"provider": cfg.agent_b.provider if cfg.agent_b else cfg.agent_a.provider, "model": (cfg.agent_b.model if cfg.agent_b else cfg.agent_a.model), "persona": (cfg.agent_b.persona if cfg.agent_b else cfg.agent_a.persona)},
    }


def _hook_snapshot(logger: RunLogger, theme_id: int) -> None:
    """Placeholder — diisi di Tahap 5 (Questionnaire)."""
    return


def run_single_run(
    config: ExperimentConfig,
    themes: list[Any],
    system_prompt: str,
    client_a: Any,
    client_b: Any,
    existing_data: dict[str, Any] | None = None,
    run_id: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    """Run satu percakapan + log. Return run_id."""
    log_dir = config.runtime.log_dir
    batch_name = config.batch_name
    models_info = _get_models_info(config)
    condition = {"research_question": config.research_question, "persona_condition": config.persona_condition, "size_group": models_info["agent_a"].get("size_group"), "family": models_info["agent_a"].get("family")}
    prompts = {"conversation_system": system_prompt, "questionnaire_system_template": ""}

    # Resume: existing_data dari partial
    if existing_data and existing_data.get("conversation"):
        run_id_existing = existing_data.get("run_id")
        logger = RunLogger(log_dir=log_dir, batch_name=batch_name, run_id=run_id_existing)
        logger.data = existing_data
        logger._started_monotonic = None
        # Re-attach callbacks: progress untuk sisa tema
        existing = list(existing_data["conversation"])
        def on_done(theme_id: int, rec: dict[str, Any]) -> None:
            logger.add_theme_record(rec)
            _hook_snapshot(logger, theme_id)
            if on_progress:
                on_progress(theme_id)
        try:
            run_conversation(client_a, client_b, themes, system_prompt, temperature=config.conversation.temperature, max_tokens=config.conversation.max_tokens, on_theme_done=on_done, existing_records=existing)
            logger.finish()
        except BaseException as exc:
            logger.fail(exc=exc, status="failed" if not isinstance(exc, KeyboardInterrupt) else "incomplete")
            raise
        return logger.run_id

    logger = RunLogger(log_dir=log_dir, batch_name=batch_name, run_id=run_id)
    config_snapshot = config.model_dump()
    logger.start(config_snapshot=config_snapshot, models_info=models_info, prompts=prompts, condition=condition)

    def on_done(theme_id: int, rec: dict[str, Any]) -> None:
        logger.add_theme_record(rec)
        _hook_snapshot(logger, theme_id)
        if on_progress:
            on_progress(theme_id)

    try:
        run_conversation(client_a, client_b, themes, system_prompt, temperature=config.conversation.temperature, max_tokens=config.conversation.max_tokens, on_theme_done=on_done)
        logger.finish()
    except BaseException as exc:
        # Jangan menghapus partial; mention in logic fail
        is_interrupt = isinstance(exc, KeyboardInterrupt)
        logger.fail(exc=exc, status="incomplete" if is_interrupt else "failed")
        raise
    return logger.run_id


def run_batch(
    config: ExperimentConfig,
    quick: bool = False,
    resume_dir: str | Path | None = None,
    overwrite: bool = False,
    client_factory: Callable[[Any], Any] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> list[str]:
    """Loop n_runs, handle concurrency, resume, quick.

    client_factory(agent_cfg) -> client (for tests). Jika None, buat LLMClient.
    on_progress(run_idx, theme_id, total_themes).
    """
    if overwrite and Path(config.runtime.log_dir, config.batch_name).exists():
        import shutil
        shutil.rmtree(Path(config.runtime.log_dir, config.batch_name))

    themes = _load_themes(config.themes_file)
    system_prompt = _load_system_prompt(config)

    if quick:
        config = config.model_copy()
        # Jangan mutasi original; quick hanya 1 run
        config.n_runs = 1

    # Kumpulkan partial untuk resume
    resume_partials: list[dict[str, Any]] = []
    search_dir = Path(resume_dir) if resume_dir is not None else Path(config.runtime.log_dir, config.batch_name)
    if resume_dir is not None or (quick is False and search_dir.exists()):
        for p in sorted(search_dir.glob("*.partial.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    resume_partials.append(json.load(f))
            except Exception:
                continue

    def make_client(agent_cfg: Any) -> Any:
        if client_factory is not None:
            return client_factory(agent_cfg)
        return LLMClient(model=agent_cfg.model, api_key_env=agent_cfg.api_key_env, base_url=agent_cfg.base_url, max_retries=config.runtime.max_retries)

    run_ids: list[str] = []

    # Jika resume, selesaikan dulu partial yang belum 36
    if resume_partials:
        # resume hanya untuk partial yang conversation < 36
        to_resume = [d for d in resume_partials if len(d.get("conversation", [])) < 36]
        for data in to_resume:
            ca = make_client(config.agent_a)
            cb = make_client(config.agent_b)  # type: ignore[arg-type]
            idx = 0
            rid = run_single_run(config, themes, system_prompt, ca, cb, existing_data=data, on_progress=lambda tid, idx=idx: on_progress(idx, tid, len(themes)) if on_progress else None)
            run_ids.append(rid)

        remaining = config.n_runs - len(run_ids)
        if remaining <= 0:
            BatchSummary(config.runtime.log_dir, config.batch_name).generate()
            return run_ids
        # Lanjut sisa new runs
        config_remaining = config.model_copy(update={"n_runs": remaining})
    else:
        config_remaining = config

    # New runs
    def do_one(idx: int) -> str:
        ca = make_client(config_remaining.agent_a)
        cb = make_client(config_remaining.agent_b)  # type: ignore[arg-type]
        return run_single_run(config_remaining, themes, system_prompt, ca, cb, on_progress=lambda tid: on_progress(idx, tid, len(themes)) if on_progress else None)

    if config_remaining.runtime.concurrency > 1 and not client_factory:
        # Real run concurrency
        with ThreadPoolExecutor(max_workers=config_remaining.runtime.concurrency) as ex:
            futs = {ex.submit(do_one, i): i for i in range(config_remaining.n_runs)}
            for fut in as_completed(futs):
                run_ids.append(fut.result())
    else:
        for i in range(config_remaining.n_runs):
            run_ids.append(do_one(i))

    BatchSummary(config.runtime.log_dir, config.batch_name).generate()
    return run_ids
