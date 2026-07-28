#!/usr/bin/env python3
"""Sync API keys from the sibling numerai repo into this one.

Copies the keys below from sibling repos' ``.env`` files into ``./.env``
(existing non-empty local values win), then mirrors them into the ``env`` block of
``.claude/settings.local.json`` so Claude Code exports them into the session
environment and ``${VAR}`` references in ``.mcp.json`` resolve.

Secret values are never printed. Run via ``make env`` (part of ``make install``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KEYS = [
    "EXA_API_KEY",
    "NUMERAI_MCP_AUTH",
    "NUMERAI_PUBLIC_ID",
    "NUMERAI_SECRET_KEY",
    "NOSIBLE_API_KEY",
    "HF_TOKEN",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ENVS = [
    Path(os.environ.get("NUMERAI_REPO", REPO_ROOT.parent / "numerai")) / ".env",
    Path(os.environ.get("NOSIBLE_REPO", REPO_ROOT.parent / "nosible-py")) / ".env",
]
LOCAL_ENV = REPO_ROOT / ".env"
CLAUDE_LOCAL_SETTINGS = REPO_ROOT / ".claude" / "settings.local.json"


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def main() -> int:
    source: dict[str, str] = {}
    for env_path in SOURCE_ENVS:
        for key, value in parse_env(env_path).items():
            if value:
                source.setdefault(key, value)
    local = parse_env(LOCAL_ENV)

    merged = {k: local.get(k) or source.get(k, "") for k in KEYS}
    missing = [k for k, v in merged.items() if not v]

    lines = ["# Synced from sibling repos by scripts/sync_env.py — gitignored."]
    lines += [f"{k}={v}" for k, v in merged.items()]
    LOCAL_ENV.write_text("\n".join(lines) + "\n")
    LOCAL_ENV.chmod(0o600)

    settings: dict = {}
    if CLAUDE_LOCAL_SETTINGS.exists():
        settings = json.loads(CLAUDE_LOCAL_SETTINGS.read_text())
    settings.setdefault("env", {}).update({k: v for k, v in merged.items() if v})
    CLAUDE_LOCAL_SETTINGS.parent.mkdir(exist_ok=True)
    CLAUDE_LOCAL_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")
    CLAUDE_LOCAL_SETTINGS.chmod(0o600)

    synced = [k for k in KEYS if merged[k]]
    print(f"synced {len(synced)} key(s) into .env and .claude/settings.local.json")
    if missing:
        print(f"warning: no value found for: {', '.join(missing)}", file=sys.stderr)
        for env_path in SOURCE_ENVS:
            if not env_path.exists():
                print(f"warning: source env not found: {env_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
