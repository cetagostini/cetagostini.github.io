#!/usr/bin/env python3
"""Verify that the alchemize_pytensor_mlx_gemma_3n conda env exists
and its Jupyter kernelspec is registered.

Called as a preflight by render-all.sh before any render pass.
Exits nonzero with an actionable message on failure.
"""

import json
import subprocess
import sys

ENV = "alchemize_pytensor_mlx_gemma_3n"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # 1. Conda env exists
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail("conda env list failed — is conda installed?")

    envs = json.loads(result.stdout)
    env_paths = envs.get("envs", [])
    found = any(p.rstrip("/").endswith(f"/{ENV}") or p.rstrip("/").endswith(f"\\{ENV}")
                for p in env_paths)
    if not found:
        fail(
            f"conda env '{ENV}' not found.\n"
            f"  Create it:  bash scripts/setup_envs.sh\n"
            f"  Or manually: conda create -n {ENV} --clone cetagostini_site"
        )

    # 2. Jupyter kernelspec registered
    result = subprocess.run(
        ["conda", "run", "-n", ENV, "jupyter", "kernelspec", "list", "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail(
            f"Cannot run jupyter in env '{ENV}'.\n"
            f"  Install ipykernel: conda run -n {ENV} pip install ipykernel\n"
            f"  Register kernel:   conda run -n {ENV} python -m ipykernel install "
            f"--user --name {ENV} --display-name 'Python ({ENV})'"
        )

    kernels = json.loads(result.stdout)
    specs = kernels.get("kernelspecs", {})
    if ENV not in specs:
        fail(
            f"Jupyter kernel '{ENV}' is not registered.\n"
            f"  Register it: conda run -n {ENV} python -m ipykernel install "
            f"--user --name {ENV} --display-name 'Python ({ENV})'"
        )

    print(f"✓ conda env '{ENV}' exists and kernel is registered")


if __name__ == "__main__":
    main()