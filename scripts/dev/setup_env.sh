#!/usr/bin/env bash
# Setup script to ensure environment is configured with p10k and icons
set -euo pipefail

log() { echo "[setup] $*"; }

# Check if we're on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
  log "This setup is for macOS. Skipping font installation."
  exit 0
fi

# Install Nerd Font if not already installed
if ! brew list font-meslo-lg-nerd-font &>/dev/null; then
  log "Installing MesloLG Nerd Font for terminal icons..."
  brew install --cask font-meslo-lg-nerd-font
else
  log "MesloLG Nerd Font already installed ✓"
fi

# Install Powerlevel10k if not already installed
if ! brew list powerlevel10k &>/dev/null; then
  log "Installing Powerlevel10k prompt..."
  brew install powerlevel10k
else
  log "Powerlevel10k already installed ✓"
fi

# Check if zshrc is configured
if grep -q "powerlevel10k.zsh-theme" ~/.zshrc; then
  log "Powerlevel10k already configured in ~/.zshrc ✓"
else
  log "Note: Run ~/env/install.sh to configure Powerlevel10k in your shell"
fi

log "Setup complete! Remember to:"
log "  1. Set your terminal font to 'MesloLGS Nerd Font' or 'MesloLGS NF'"
log "  2. Restart your terminal or run: exec zsh"