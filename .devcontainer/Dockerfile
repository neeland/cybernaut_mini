# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    jq \
    unzip \
    zip \
    wget \
    ca-certificates \
    sqlite3 \
    libgomp1 \
    graphviz \
    zsh \
    tmux \
    neovim \
    btop \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS (system-wide) — required for npx-based MCP servers, Claude Code
# plugins, and the oh-my-claudecode (OMC) HUD.
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code + oh-my-claudecode CLI (system-wide) — baked into the IMAGE so
# `claude` and `omc` exist even if post-create fails or is skipped. post-create
# (via scripts/setup_claude_omc.sh) updates both to latest and wires the OMC
# plugin/HUD/hooks; the npm prefix is root-owned so the settings merge disables
# claude's self-updater (updates go through the setup script instead).
RUN npm install -g @anthropic-ai/claude-code@latest oh-my-claude-sisyphus@latest \
    && command -v claude && command -v omc

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/home/vscode/.local/bin:${PATH}"

# Modern CLI niceties (shared with ~/env). apt-provided:
RUN apt-get update && apt-get install -y --no-install-recommends \
    fzf bat fd-find ripgrep zoxide \
    zsh-autosuggestions zsh-syntax-highlighting \
    && rm -rf /var/lib/apt/lists/*

# Data CLI tooling:
#   duckdb : SQL straight over the corpus/artifacts without Python —
#            e.g. duckdb -c "select count(*) from 'data/sample/documents.jsonl'"
#   yq     : query/validate the YAML configs (configs/*.yaml, conf/base/*)
#            from the shell, jq-style.
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) duck="amd64"; yqa="amd64" ;; \
      arm64) duck="arm64"; yqa="arm64" ;; \
      *) duck=""; yqa="" ;; \
    esac; \
    if [ -n "$duck" ]; then \
      tmp="$(mktemp -d)"; \
      curl -fsSL "https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-${duck}.zip" -o "$tmp/duckdb.zip"; \
      unzip -q "$tmp/duckdb.zip" -d "$tmp"; \
      install -m755 "$tmp/duckdb" /usr/local/bin/duckdb; \
      curl -fsSL "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${yqa}" -o /usr/local/bin/yq; \
      chmod +x /usr/local/bin/yq; \
      rm -rf "$tmp"; \
    else \
      echo "WARN: no duckdb/yq build for arch=$arch"; \
    fi

# eza / delta / dust (not in Debian stable) + starship prompt.
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) t="x86_64-unknown-linux-gnu" ;; \
      arm64) t="aarch64-unknown-linux-gnu" ;; \
      *) t="" ;; \
    esac; \
    if [ -n "$t" ]; then \
      tmp="$(mktemp -d)"; \
      curl -fsSL "https://github.com/eza-community/eza/releases/latest/download/eza_${t}.tar.gz" | tar -xz -C "$tmp"; \
      curl -fsSL "https://github.com/dandavison/delta/releases/download/0.18.2/delta-0.18.2-${t}.tar.gz" | tar -xz -C "$tmp"; \
      curl -fsSL "https://github.com/bootandy/dust/releases/download/v1.1.1/dust-v1.1.1-${t}.tar.gz" | tar -xz -C "$tmp"; \
      install -m755 "$(find "$tmp" -type f -name eza   | head -1)" /usr/local/bin/eza; \
      install -m755 "$(find "$tmp" -type f -name delta | head -1)" /usr/local/bin/delta; \
      install -m755 "$(find "$tmp" -type f -name dust  | head -1)" /usr/local/bin/dust; \
      rm -rf "$tmp"; \
    fi; \
    curl -sS https://starship.rs/install.sh | sh -s -- -y

RUN python -m pip install --no-cache-dir --upgrade pip
