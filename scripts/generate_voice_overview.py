#!/usr/bin/env python3
"""Generate voice overviews of articles using MiMo V2.5 TTS VoiceDesign.

Uses the mimo-v2.5-tts-voicedesign model to convert article summaries
into natural speech with configurable emotional tone.

Model: https://mimo.mi.com/models/en-US/mimo-v2.5-tts-voicedesign
API:   OpenAI-compatible at https://api.xiaomimimo.com/v1

Requires:
    pip install openai python-frontmatter
    export MIMO_API_KEY=...   # or set in .env

Usage:
    # Generate all emotions for an article
    python scripts/generate_voice_overview.py articles/placebo_bayesian_quasi_experiments/placebo_bayesian_quasi_experiments.qmd

    # Generate a specific emotion
    python scripts/generate_voice_overview.py articles/my_article/my_article.qmd --emotion enthusiastic

    # Custom voice description (director mode)
    python scripts/generate_voice_overview.py articles/my_article/my_article.qmd \\
        --voice "A warm, contemplative female voice, speaking slowly with pauses for emphasis"

    # List available articles
    python scripts/generate_voice_overview.py --list

    # List available emotion presets
    python scripts/generate_voice_overview.py --emotions
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

try:
    import frontmatter
except ImportError:
    frontmatter = None

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "audio" / "voice-overviews"
ARTICLES_DIR = ROOT / "articles"

# ── Emotion presets ────────────────────────────────────────────────────────────
# Each preset is a natural-language voice description sent in the `user` message
# of the mimo-v2.5-tts-voicedesign API. These use the "director mode" style:
# character + scene + guidance.

EMOTION_PRESETS: dict[str, str] = {
    "enthusiastic": (
        "A bright, energetic voice full of genuine excitement — like a researcher who just "
        "had a breakthrough and can't wait to share it. Speaking at a brisk pace with rising "
        "intonation on key points, slight breathlessness from enthusiasm, and a warm, "
        "inviting tone that pulls the listener in."
    ),
    "thoughtful": (
        "A calm, measured voice with the unhurried cadence of deep reflection — like a "
        "seasoned professor thinking out loud in a quiet study. Moderate pace with deliberate "
        "pauses between ideas, a warm lower register, and the subtle weight of someone who "
        "has spent years wrestling with these questions."
    ),
    "storyteller": (
        "A warm, engaging narrative voice — like a skilled podcast host drawing you into a "
        "fascinating story. Natural conversational rhythm with varied pacing: slower for "
        "setup, quicker through action, pausing for dramatic effect before key reveals. "
        "Friendly and accessible, never condescending."
    ),
    "confident": (
        "A clear, authoritative voice with quiet confidence — like a practitioner presenting "
        "results they trust completely. Steady pacing, no hedging, a direct and grounded tone "
        "that conveys mastery without arrogance. Each sentence lands with conviction."
    ),
    "curious": (
        "An inquisitive, slightly playful voice — like a detective who just noticed something "
        "interesting. Rising inflection on questions, a conspiratorial lean-in on surprising "
        "findings, and the genuine wonder of someone who finds joy in discovery. Moderate pace "
        "with animated emphasis."
    ),
    "serious": (
        "A grave, focused voice with the weight of important stakes — like a doctor explaining "
        "a diagnosis. Slow, deliberate pacing with clear articulation. No filler, no jokes. "
        "The tone conveys that what follows matters and deserves full attention."
    ),
}


# ── Article extraction ─────────────────────────────────────────────────────────

def strip_quarto_markup(text: str) -> str:
    """Remove Quarto/Pandoc markup, leaving clean prose."""
    # Remove YAML frontmatter delimiters
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    # Remove code cells (#| directives + code blocks)
    text = re.sub(r"```\{.*?\}\n.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove images and links markup
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    # Remove LaTeX display math
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    # Remove LaTeX inline math
    text = re.sub(r"\$[^$]+\$", "", text)
    # Remove Quarto callout divs
    text = re.sub(r":::\{.*?\}", "", text)
    text = re.sub(r":::", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown headers (keep the text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_article_content(qmd_path: Path) -> dict:
    """Extract title, description, and body text from a .qmd file.

    Returns dict with keys: title, description, authors, date, categories, body.
    """
    raw = qmd_path.read_text(encoding="utf-8")

    # Parse frontmatter if available
    meta: dict = {}
    if frontmatter is not None:
        try:
            post = frontmatter.loads(raw)
            meta = dict(post.metadata)
            body_raw = post.content
        except Exception:
            body_raw = raw
    else:
        # Manual YAML extraction
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if fm_match:
            body_raw = raw[fm_match.end():]
            for line in fm_match.group(1).split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip().strip('"').strip("'")
        else:
            body_raw = raw

    body = strip_quarto_markup(body_raw)

    # Truncate to ~1500 chars (roughly 250 words, ~90s of speech)
    if len(body) > 1500:
        # Cut at last sentence boundary before limit
        truncated = body[:1500]
        last_period = truncated.rfind(".")
        if last_period > 800:
            body = truncated[: last_period + 1]
        else:
            body = truncated + "..."

    return {
        "title": meta.get("title", qmd_path.stem.replace("_", " ").title()),
        "description": meta.get("description", ""),
        "authors": meta.get("author", "Carlos Trujillo"),
        "date": meta.get("date", ""),
        "categories": meta.get("categories", []),
        "body": body,
    }


def build_overview_text(article: dict) -> str:
    """Build a spoken overview from extracted article content."""
    parts: list[str] = []

    # Opening
    parts.append(f"Article: {article['title']}.")

    if article["description"]:
        parts.append(article["description"])

    # Body excerpt
    parts.append(article["body"])

    # Closing
    if article["categories"]:
        cats = ", ".join(article["categories"][:5])
        parts.append(f"Topics covered: {cats}.")

    return " ".join(parts)


# ── TTS generation ─────────────────────────────────────────────────────────────

def build_client() -> OpenAI:
    """Build the MiMo API client."""
    # Try .env file first
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        print(
            "Error: MIMO_API_KEY not set.\n"
            "  Set it in .env or export MIMO_API_KEY=<your-key>\n"
            "  Get a key at: https://platform.xiaomimimo.com/console/balance",
            file=sys.stderr,
        )
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url="https://api.xiaomimimo.com/v1")


def generate_audio(
    client: OpenAI,
    text: str,
    voice_description: str,
    output_path: Path,
) -> Path:
    """Generate audio from text using MiMo V2.5 TTS VoiceDesign.

    Args:
        client: OpenAI-compatible client pointing at MiMo API.
        text: The text to synthesize (goes in assistant message).
        voice_description: Natural-language voice/style description (goes in user message).
        output_path: Where to write the .wav file.

    Returns:
        Path to the generated audio file.
    """
    messages = [
        {"role": "user", "content": voice_description},
        {"role": "assistant", "content": text},
    ]

    completion = client.chat.completions.create(
        model="mimo-v2.5-tts-voicedesign",
        messages=messages,
        audio={"format": "wav"},
    )

    message = completion.choices[0].message
    if message.audio is None or not getattr(message.audio, "data", None):
        print("Error: No audio data returned from API.", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_bytes = base64.b64decode(message.audio.data)
    output_path.write_bytes(audio_bytes)

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def find_articles() -> list[Path]:
    """Find all .qmd article files."""
    articles: list[Path] = []
    if ARTICLES_DIR.exists():
        for article_dir in sorted(ARTICLES_DIR.iterdir()):
            if article_dir.is_dir():
                qmd_files = list(article_dir.glob("*.qmd"))
                articles.extend(qmd_files)
    return articles


def slug_from_path(path: Path) -> str:
    """Derive a slug from the article path."""
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate voice overviews of articles using MiMo V2.5 TTS VoiceDesign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s articles/placebo_bayesian_quasi_experiments/placebo_bayesian_quasi_experiments.qmd
              %(prog)s articles/my_article/my_article.qmd --emotion enthusiastic
              %(prog)s articles/my_article/my_article.qmd --emotion enthusiastic --emotion thoughtful
              %(prog)s articles/my_article/my_article.qmd --voice "A deep, gravelly male voice..."
              %(prog)s --list
              %(prog)s --emotions
        """),
    )
    parser.add_argument(
        "article",
        nargs="?",
        help="Path to a .qmd article file",
    )
    parser.add_argument(
        "--emotion",
        "-e",
        action="append",
        choices=list(EMOTION_PRESETS.keys()),
        help="Emotion preset(s) to use. Can specify multiple. Default: all presets.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        help="Custom voice description (overrides --emotion). Uses director mode.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for audio files. Default: {OUTPUT_DIR.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available articles and exit.",
    )
    parser.add_argument(
        "--emotions",
        action="store_true",
        help="List available emotion presets and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── List modes ─────────────────────────────────────────────────────────────
    if args.emotions:
        print("Available emotion presets:\n")
        for name, desc in EMOTION_PRESETS.items():
            wrapped = textwrap.fill(desc, width=72, initial_indent="  ", subsequent_indent="  ")
            print(f"  {name}")
            print(wrapped)
            print()
        return

    if args.list:
        articles = find_articles()
        if not articles:
            print("No articles found in articles/")
            return
        print("Available articles:\n")
        for a in articles:
            slug = slug_from_path(a)
            rel = a.relative_to(ROOT)
            print(f"  {slug}")
            print(f"    {rel}")
        print()
        return

    # ── Validate input ─────────────────────────────────────────────────────────
    if not args.article:
        parser = argparse.ArgumentParser()
        parser.print_help()
        print("\nError: article path is required (unless using --list or --emotions)", file=sys.stderr)
        sys.exit(1)

    article_path = Path(args.article)
    if not article_path.is_absolute():
        article_path = ROOT / article_path
    if not article_path.exists():
        print(f"Error: Article not found: {article_path}", file=sys.stderr)
        sys.exit(1)

    # ── Extract content ────────────────────────────────────────────────────────
    article = extract_article_content(article_path)
    overview_text = build_overview_text(article)
    slug = slug_from_path(article_path)

    print(f"Article: {article['title']}")
    print(f"Slug:    {slug}")
    print(f"Text:    {len(overview_text)} chars (~{len(overview_text) // 6} words)")
    print()

    # ── Determine voice descriptions ───────────────────────────────────────────
    if args.voice:
        voices = {"custom": args.voice}
    elif args.emotion:
        voices = {e: EMOTION_PRESETS[e] for e in args.emotion}
    else:
        voices = dict(EMOTION_PRESETS)

    # ── Generate ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("Dry run — would generate:")
        for emotion_name in voices:
            out = args.output_dir / slug / f"{slug}_{emotion_name}.wav"
            print(f"  {emotion_name}: {out.relative_to(ROOT)}")
        return

    client = build_client()
    output_dir = args.output_dir / slug

    for emotion_name, voice_desc in voices.items():
        out_path = output_dir / f"{slug}_{emotion_name}.wav"
        print(f"Generating [{emotion_name}]...")
        try:
            result = generate_audio(client, overview_text, voice_desc, out_path)
            size_kb = result.stat().st_size / 1024
            print(f"  ✓ {result.relative_to(ROOT)}  ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
