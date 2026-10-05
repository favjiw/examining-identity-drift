from idrift.config import ExperimentConfig, load_config

def test_default_agent_b_same_as_agent_a():
    cfg = ExperimentConfig(batch_name="test", n_runs=1, agent_a={"provider": "groq", "model": "llama-70b"})
    assert cfg.agent_b is not None
    assert cfg.agent_b.provider == "groq"
    assert cfg.agent_b.model == "llama-70b"

def test_agent_b_same_as_explicit():
    cfg = ExperimentConfig(batch_name="test", agent_b={"same_as": "agent_a"})
    assert cfg.agent_b is not None
    assert cfg.agent_b.model == "gpt-4o" # agent_a default

def test_load_yaml(tmp_path):
    yml = """
batch_name: rq1_llama70b
n_runs: 5
conversation:
  temperature: 0.8
agent_a:
  provider: together
  model: llama-3.1-70b
    """
    p = tmp_path / "cfg.yaml"
    p.write_text(yml)
    cfg = load_config(p)
    assert cfg.batch_name == "rq1_llama70b"
    assert cfg.n_runs == 5
    assert cfg.conversation.temperature == 0.8
    assert cfg.agent_b.provider == "together"
