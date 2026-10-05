"""Shared small helpers (Tahap 1: placeholder)."""


def short_uuid(n: int = 6) -> str:
    """Short random hex id for run_id."""
    import secrets

    return secrets.token_hex(n // 2 + 1)[:n]
