"""Config models + YAML loader. API key never stored here, only env name."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ConversationConfig(BaseModel):
    """Conversation call params (PRD §3.1)."""

    temperature: float = 0.7
    system_prompt_file: str = "data/prompts/conversation_system.txt"
    max_tokens: int = 512


class QuestionnaireConfig(BaseModel):
    """Questionnaire params (PRD §3.4)."""

    temperature: float = 0
    repeats: int = 10
    snapshots: list[int] = Field(default_factory=lambda: [12, 24, 36])
    measure_agents: list[Literal["A", "B"]] = Field(default_factory=lambda: ["A", "B"])
    include: str | list[str] = "all"


class AgentConfig(BaseModel):
    """One agent endpoint. OpenAI-compatible (PRD §3.2)."""

    provider: str = "openai"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o"
    size_group: Literal["small", "medium", "large"] | None = None
    family: str | None = None
    persona: str | None = None
    same_as: str | None = None


class RuntimeConfig(BaseModel):
    """Runtime knobs (PRD FR-1, §8)."""

    concurrency: int = 2
    max_retries: int = 5
    log_dir: str = "logs"


class ExperimentConfig(BaseModel):
    """Top-level batch config (PRD §5)."""

    batch_name: str
    research_question: Literal["RQ1", "RQ2"] = "RQ1"
    n_runs: int = 20
    themes_file: str = "data/themes.json"
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    questionnaire: QuestionnaireConfig = Field(default_factory=QuestionnaireConfig)
    agent_a: AgentConfig = Field(default_factory=AgentConfig)
    agent_b: AgentConfig | None = None
    persona_condition: Literal["none", "low", "high"] = "none"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _resolve_agent_b(self) -> ExperimentConfig:
        """Default agent_b same as agent_a (paper: same model both sides)."""
        if self.agent_b is None:
            self.agent_b = AgentConfig(
                provider=self.agent_a.provider,
                base_url=self.agent_a.base_url,
                api_key_env=self.agent_a.api_key_env,
                model=self.agent_a.model,
                size_group=self.agent_a.size_group,
                family=self.agent_a.family,
                persona=self.agent_a.persona,
            )
        elif self.agent_b.same_as == "agent_a":
            a = self.agent_a.model_dump()
            a.pop("same_as", None)
            self.agent_b = AgentConfig(**a)
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML file into ExperimentConfig."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
