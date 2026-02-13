#!/bin/bash

# Script to rebuild and preview Quarto site
#
# Usage:
#   bash quarto-rebuild.sh            # Render from freeze + preview
#   bash quarto-rebuild.sh --clean    # Wipe freeze/cache, re-execute everything, then preview

set -e

if [[ "$1" == "--clean" ]]; then
    echo "Removing Quarto cache and freeze..."
    rm -rf .quarto/
    rm -rf _freeze/
fi

echo "Rendering Quarto site..."
quarto render

echo "Starting Quarto preview server..."
quarto preview
