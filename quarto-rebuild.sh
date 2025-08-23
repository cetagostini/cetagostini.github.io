#!/bin/bash

# Script to rebuild and preview Quarto site
# 1. Remove cache and freeze
# 2. Render the site
# 3. Preview the site

echo "Removing Quarto cache and freeze..."
rm -rf .quarto/
rm -rf _freeze/

echo "Rendering Quarto site..."
quarto render

echo "Starting Quarto preview server..."
quarto preview