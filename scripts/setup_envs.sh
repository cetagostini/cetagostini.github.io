#!/bin/bash
# Provision the conda environments + Jupyter kernels for the site and every article.
#
#   bash scripts/setup_envs.sh              # create missing envs, (re-)register kernels
#   bash scripts/setup_envs.sh --recreate   # delete + recreate every env from its yml
#
# Layout (one env per yml, env name == kernel name):
#   environment.yml                   -> cetagostini_site  (Quarto project engine,
#                                          post-render scripts, pages without a kernel)
#   articles/<slug>/environment.yml   -> <slug>             (article's own kernel,
#                                          selected by `jupyter: <slug>` frontmatter)
#
# Idempotent: existing envs are left untouched unless --recreate.
# Needs conda (mamba used when available). ~10 GB of envs total on first run.

set -euo pipefail
cd "$(dirname "$0")/.."

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

CREATE="conda"
command -v mamba >/dev/null 2>&1 && CREATE="mamba"

env_exists() { conda env list | awk 'NR>2 {print $1}' | grep -qx "$1"; }

setup_env() {
    local name="$1" yml="$2"
    if [[ $RECREATE == 1 ]] && env_exists "$name"; then
        echo "  - removing $name"
        conda env remove -n "$name" -y >/dev/null
    fi
    if env_exists "$name"; then
        echo "  = $name (exists)"
    else
        echo "  + $name  <-  $yml"
        $CREATE env create -n "$name" -f "$yml"
    fi
    conda run -n "$name" python -m ipykernel install --user \
        --name "$name" --display-name "Python ($name)" >/dev/null
    echo "  k kernel '$name' registered"
}

echo "Base env (site engine):"
setup_env cetagostini_site environment.yml

echo "Article envs:"
for yml in articles/*/environment.yml; do
    slug="$(basename "$(dirname "$yml")")"
    setup_env "$slug" "$yml"
done

echo
echo "Done. Check with: jupyter kernelspec list"
