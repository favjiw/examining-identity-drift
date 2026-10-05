# idrift — Identity Drift Replication CLI

Replication tool for *"Examining Identity Drift in Conversations of LLM Agents"* (Choi et al., arXiv:2412.00804).

## Setup

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -e .[dev]
```

## Quick Start (Tahap 1)

```bash
# Check version
idrift version
```

## Implementation Notes & Assumptions

- **Themes (36 items)**: Tema diambil dari Aron, Melinat, Aron, Vallone, & Bator (1997), *Personality and Social Psychology Bulletin*, 23(4), 363-377, Appendix (Task Slips for Closeness-Generating Procedure). Transkripsi manual dari PDF scan; sumber PDF disimpan di `data/sources/`.
- **API Keys**: Only read from environment variables defined in config (`api_key_env`). Never logged or exported.
- **Sequential User Messages**: Consecutive user messages in snapshot / conversation setup are concatenated with newlines per PRD §3.3 to comply with standard chat completion API role requirements.
