# `scripts/claude/`

## OAuth token setup (subscription auth)

`setup-oauth-token.sh` mints a Claude subscription OAuth token and persists it
into `.env` so the devcontainer picks it up at boot. It runs the interactive
`claude setup-token` browser flow under `script(1)`, scrapes the printed
`sk-ant-oat01-...` token from the transcript, writes it to
`CLAUDE_CODE_OAUTH_TOKEN=` in `.env` (backing `.env` up first), and blanks
`ANTHROPIC_API_KEY` (exactly one auth method may be set). Run it **on the host**
(it needs a browser); it is executed, not sourced:

```bash
./scripts/claude/setup-oauth-token.sh                       # interactive mint + persist
./scripts/claude/setup-oauth-token.sh --token sk-ant-oat01-...   # already have a token
```

You rarely need to run it by hand: `.devcontainer/sync-env.sh` (the
devcontainer `initializeCommand`) offers this flow automatically **before the
container is built** whenever `CLAUDE_CODE_OAUTH_TOKEN` is empty and a terminal
is attached (15s auto-skip; headless builds just log a note). Minting before
build means docker's `--env-file` injects the token on the first boot — no
rebuild round-trip. Inside an already-running container,
`scripts/dev/shell.sh` also injects the current `.env`, so a freshly minted
token works in those sessions immediately.

## Local-model routing

Point Claude Code at a **local LM Studio model** (or an Anthropic-compatible proxy)
instead of the Anthropic cloud — and revert back. Both scripts must be **sourced**
(they export/unset env that Claude Code reads at startup), and Claude Code must be
**restarted** afterwards (reload window / new `claude` session) to pick it up.

| Script | Does |
| --- | --- |
| `set-claude-env.sh` | export `ANTHROPIC_BASE_URL` + per-tier `ANTHROPIC_*_MODEL` overrides → local LM Studio / proxy; defines `claude_env_test` for a reachability check |
| `set-claude-env.sh reset` | unset those overrides (built-in revert path) |
| `reset-claude-env.sh` | standalone revert — unset the overrides, restore the Anthropic-cloud default |

```bash
source ./scripts/claude/set-claude-env.sh          # enable local model
claude_env_test                                    # check the endpoint is reachable
source ./scripts/claude/reset-claude-env.sh        # back to Anthropic cloud
```

Override the target before sourcing with `LMSTUDIO_BASE_URL` / `LMSTUDIO_HOST` /
`LMSTUDIO_PORT` / `LMSTUDIO_MODEL`.

## Protocol note (the #1 reason local Claude Code silently fails)

Claude Code talks Anthropic's Messages API (`POST BASE_URL/v1/messages`). LM Studio's
built-in server is OpenAI-compatible (`/v1/chat/completions`) and does **not** serve
`/v1/messages` — pointing `BASE_URL` straight at it usually 404s. Put a translation
proxy in front (claude-code-router or LiteLLM) and set `BASE_URL` to the proxy.

## Context-length note

Claude Code's initial prompt (system prompt + tool catalog) is large. Small local
models loaded with a short context window 500 with *"tokens to keep … greater than
the context length."* Mitigate by loading the model with a bigger **Context Length**
(≈16384) plus **Q8 K/V cache quantization**, and start the slim session with
`claude --strict-mcp-config --mcp-config '{"mcpServers":{}}'` to drop MCP servers
from the prompt.
