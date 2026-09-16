#!/usr/bin/env python3
"""Re-export the live conda envs into their environment.yml files.

Run this after installing/removing packages in an article's env so the
committed yml stays the source of truth:

    python3 scripts/export_envs.py                              # base + every article
    python3 scripts/export_envs.py cross_city_media_spillovers  # just one env

Machine-specific content is stripped automatically:
  - ``prefix:`` lines
  - editable installs (``-e``, e.g. the repo's own ``cetagostini`` package)
  - local-path pip installs (``file:///...``)

Direct ``git+https`` pip installs are preserved as URLs: ``conda env export``
flattens them into version pins that would not rebuild the same tree
elsewhere, so they are restored from ``pip freeze``.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_HEADER = """\
# Base environment for the site itself (cetagostini_site).
#
# Quarto project engine (execute.conda in _quarto.yml), post-render scripts
# (js/build-llms-md.py, generate_sitemap.py, generate_articles_network.py),
# and pages without their own kernel (index, about, diary, talks, listings).
# Articles do NOT run here: each articles/<slug>/ has its own environment.yml
# and kernel, selected by "jupyter: <slug>" in the article frontmatter.
#
# Regenerate from the live env: python3 scripts/export_envs.py cetagostini_site
"""

ARTICLE_HEADER = """\
# Environment for the "{slug}" article (articles/{slug}/).
# Kernel name == env name == article slug; selected by "jupyter: {slug}"
# in the article frontmatter. Provision with: bash scripts/setup_envs.sh
# Regenerate from the live env: python3 scripts/export_envs.py {slug}
"""


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr[:2000]}")
    return proc.stdout


def pip_freeze_maps(env: str) -> tuple[dict[str, str], set[str]]:
    """(git URL by normalized pkg name, editable pkg names) from pip freeze."""
    git_urls: dict[str, str] = {}
    editable: set[str] = set()
    for line in run(["conda", "run", "-n", env, "pip", "freeze"]).splitlines():
        line = line.strip()
        if line.startswith("-e "):
            m = re.search(r"#egg=([A-Za-z0-9._-]+)", line)
            if m:
                editable.add(norm(m.group(1)))
            continue
        if " @ git+" in line:
            pkg, url = line.split(" @ ", 1)
            git_urls[norm(pkg)] = url.strip()
    return git_urls, editable


def norm(pkg: str) -> str:
    return pkg.strip().lower().replace("_", "-")


def filter_export(env: str, exported: str) -> str:
    git_urls, editable = pip_freeze_maps(env)
    out: list[str] = []
    for line in exported.splitlines():
        if line.startswith("prefix:"):
            continue
        s = line.strip()
        if s.startswith("- -e ") or "file:///" in line:
            continue
        # pip-section pins look like "      - pkg==1.2.3" (indented, "- " prefix)
        m = re.match(r"^(\s+- )([A-Za-z0-9._-]+)==\S+$", line)
        if m:
            key = norm(m.group(2))
            if key in editable:
                continue
            if key in git_urls:
                out.append(f"{m.group(1)}{m.group(2)} @ {git_urls[key]}")
                continue
        out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


def export_one(env: str, yml: Path) -> None:
    exported = run(["conda", "env", "export", "-n", env, "--no-builds"])
    header = BASE_HEADER if env == "cetagostini_site" else ARTICLE_HEADER.format(slug=env)
    yml.write_text(header + filter_export(env, exported), encoding="utf-8")
    print(f"  exported {env} -> {yml.relative_to(ROOT)}")


def main() -> None:
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = ["cetagostini_site"] + sorted(
            p.parent.name for p in (ROOT / "articles").glob("*/*.qmd")
        )
    for env in targets:
        yml = ROOT / "environment.yml" if env == "cetagostini_site" else ROOT / f"articles/{env}/environment.yml"
        export_one(env, yml)


if __name__ == "__main__":
    main()
