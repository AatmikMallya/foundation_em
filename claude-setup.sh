#!/bin/bash
# Claude Code setup script for compute cluster
# Run this script with: source ./claude-setup.sh

# Load required modules
module load nodejs/18.12.1-GCCcore-12.2.0

# Set up PATH to include nodejs and npm-global bin directories
export PATH=/gpfs/radev/apps/avx512/software/nodejs/18.12.1-GCCcore-12.2.0/bin:~/.npm-global/bin:/bin:/usr/bin:/usr/local/bin:$PATH

# Set npm configuration
export npm_config_prefix=~/.npm-global

echo "Claude Code environment loaded successfully!"
echo "You can now run: claude --version"
echo "To start Claude Code in your project directory: cd your-project && claude" 