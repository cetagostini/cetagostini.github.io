#!/usr/bin/env python3
"""Generate voice overviews of articles using MiMo V2.5 TTS.

Two models are available:

  mimo-v2.5-tts             — Built-in voices (Chloe, Mia, Milo, etc.) with audio tag control.
                               Embed [crying], [pause], [sigh], etc. directly in the text.
  mimo-v2.5-tts-voicedesign — Generate any voice from a text description (director mode).
                               No presets; the user message describes the voice character.

Model docs: https://mimo.mi.com/models/en-US/mimo-v2.5-tts
            https://mimo.mi.com/models/en-US/mimo-v2.5-tts-voicedesign

Audio files live alongside each article:
    articles/<slug>/audio/<slug>_<label>.wav

Requires:
    pip install openai python-frontmatter
    export MIMO_API_KEY=...   # or set in .env

Usage:
    # Default: all emotion presets (voicedesign model)
    python scripts/generate_voice_overview.py articles/my_article/my_article.qmd

    # Built-in voice with audio tags
    python scripts/generate_voice_overview.py articles/my_article/my_article.qmd \\
        --model tts --voice-id Chloe

    # Specific emotion preset
    python scripts/generate_voice_overview.py articles/my_article/my_article.qmd -e enthusiastic

    # Custom voice description (director mode)
    python scripts/generate_voice_overview.py articles/my_article/my_article.qmd \\
        --voice "A warm, contemplative female voice, speaking slowly"

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

try:
    import frontmatter
except ImportError:
    frontmatter = None

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = ROOT / "articles"

# ── Emotion presets (voicedesign model) ────────────────────────────────────────
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

# ── Audio tag presets (mimo-v2.5-tts model) ───────────────────────────────────
# These use style tags and inline audio tags embedded in the assistant message.
# See: https://mimo.mi.com/docs/en-US/quick-start/usage-guide/audio/speech-synthesis-v2.5
#
# Format:  (style) Text with [audio tags] inline
#
# Supported audio tags (non-exhaustive):
#   [pause] [sigh] [inhale] [deep breath] [laugh] [chuckle] [sob] [cry]
#   [whisper] [shout] [trembling] [nervous] [excited] [tired] [smile]

AUDIO_TAG_PRESETS: dict[str, dict[str, str]] = {
    "warm": {
        "style": "(Gentle Warm)",
        "description": "Gentle, warm delivery with natural pauses",
    },
    "dramatic": {
        "style": "(Serious Magnetic)",
        "description": "Dramatic, magnetic delivery with emotional weight",
    },
    "energetic": {
        "style": "(Excited Lively)",
        "description": "Energetic, lively delivery with enthusiasm",
    },
    "contemplative": {
        "style": "(Calm Deep)",
        "description": "Calm, deep, contemplative delivery",
    },
    "narrative": {
        "style": "(Warm Clear)",
        "description": "Clear narrative delivery with storytelling rhythm",
    },
}

# Built-in voice IDs for mimo-v2.5-tts
BUILTIN_VOICES = ["mimo_default", "Mia", "Chloe", "Milo", "Dean"]

# ── Article extraction ─────────────────────────────────────────────────────────

def strip_quarto_markup(text: str) -> str:
    """Remove Quarto/Pandoc markup, leaving clean prose."""
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    text = re.sub(r"```\{.*?\}\n.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", "", text)
    text = re.sub(r":::\{.*?\}", "", text)
    text = re.sub(r":::", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_article_content(qmd_path: Path) -> dict:
    """Extract title, description, and body text from a .qmd file."""
    raw = qmd_path.read_text(encoding="utf-8")

    meta: dict = {}
    if frontmatter is not None:
        try:
            post = frontmatter.loads(raw)
            meta = dict(post.metadata)
            body_raw = post.content
        except Exception:
            body_raw = raw
    else:
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
    parts.append(f"Article: {article['title']}.")
    if article["description"]:
        parts.append(article["description"])
    parts.append(article["body"])
    if article["categories"]:
        cats = ", ".join(article["categories"][:5])
        parts.append(f"Topics covered: {cats}.")
    return " ".join(parts)


# ── TTS generation ─────────────────────────────────────────────────────────────

def build_client() -> OpenAI:
    """Build the MiMo API client."""
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


def generate_audio_voicedesign(
    client: OpenAI,
    text: str,
    voice_description: str,
    output_path: Path,
) -> Path:
    """Generate audio using mimo-v2.5-tts-voicedesign (voice from text description)."""
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


def generate_audio_builtin(
    client: OpenAI,
    text: str,
    style_tag: str,
    voice_id: str,
    output_path: Path,
    style_instruction: str | None = None,
) -> Path:
    """Generate audio using mimo-v2.5-tts (built-in voices + audio tags).

    Args:
        client: OpenAI-compatible client.
        text: Text to synthesize (may contain [audio tags]).
        style_tag: Opening style tag, e.g. "(Gentle Warm)".
        voice_id: Built-in voice ID (Chloe, Mia, Milo, Dean, mimo_default).
        output_path: Where to write the .wav.
        style_instruction: Optional natural-language style in user message.
    """
    messages: list[dict[str, str]] = []

    # user message is optional for mimo-v2.5-tts; use it for natural-language style
    if style_instruction:
        messages.append({"role": "user", "content": style_instruction})

    # assistant message: style tag prefix + text (with audio tags)
    messages.append({"role": "assistant", "content": f"{style_tag}{text}"})

    completion = client.chat.completions.create(
        model="mimo-v2.5-tts",
        messages=messages,
        audio={"format": "wav", "voice": voice_id},
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
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate voice overviews of articles using MiMo V2.5 TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            models:
              voicedesign (default)  Generate a voice from a text description.
                                     Uses --emotion presets or --voice custom description.
              tts                    Use built-in voices with audio tag control.
                                     Uses --voice-id (Chloe, Mia, Milo, Dean) and style tags.
                                     Supports [pause], [sigh], [cry], [laugh], etc. in text.

            examples:
              %(prog)s articles/my_article/my_article.qmd
              %(prog)s articles/my_article/my_article.qmd -e enthusiastic
              %(prog)s articles/my_article/my_article.qmd --model tts --voice-id Chloe
              %(prog)s articles/my_article/my_article.qmd --voice "A deep male voice..."
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
        "--model",
        "-m",
        choices=["voicedesign", "tts"],
        default="voicedesign",
        help="TTS model: 'voicedesign' (voice from description) or 'tts' (built-in voices). "
             "Default: voicedesign",
    )
    parser.add_argument(
        "--emotion",
        "-e",
        action="append",
        choices=list(EMOTION_PRESETS.keys()),
        help="Emotion preset(s) for voicedesign model. Can specify multiple. Default: all.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        help="Custom voice description for voicedesign model (overrides --emotion).",
    )
    parser.add_argument(
        "--voice-id",
        choices=BUILTIN_VOICES,
        default="Chloe",
        help="Built-in voice ID for tts model. Default: Chloe",
    )
    parser.add_argument(
        "--tag-preset",
        choices=list(AUDIO_TAG_PRESETS.keys()),
        default="warm",
        help="Audio tag style preset for tts model. Default: warm",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Override output directory. Default: articles/<slug>/audio/",
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
        print("Emotion presets (voicedesign model):\n")
        for name, desc in EMOTION_PRESETS.items():
            wrapped = textwrap.fill(desc, width=72, initial_indent="  ", subsequent_indent="  ")
            print(f"  {name}")
            print(wrapped)
            print()
        print("Audio tag presets (tts model):\n")
        for name, preset in AUDIO_TAG_PRESETS.items():
            print(f"  {name}")
            print(f"    Style: {preset['style']}")
            print(f"    {preset['description']}")
            print()
        print("Audio tags you can embed in text (tts model):\n")
        tags = [
            "[pause]", "[sigh]", "[inhale]", "[deep breath]",
            "[laugh]", "[chuckle]", "[sob]", "[cry]",
            "[whisper]", "[shout]", "[trembling]", "[nervous]",
            "[excited]", "[tired]", "[smile]", "[crying]",
        ]
        print(f"  {' '.join(tags)}")
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

    # Output dir: per-article audio/ unless overridden
    output_dir = args.output_dir if args.output_dir else article_path.parent / "audio"

    print(f"Article: {article['title']}")
    print(f"Slug:    {slug}")
    print(f"Model:   mimo-v2.5-tts{'-voicedesign' if args.model == 'voicedesign' else ''}")
    print(f"Text:    {len(overview_text)} chars (~{len(overview_text) // 6} words)")
    print(f"Output:  {output_dir.relative_to(ROOT)}/")
    print()

    # ── Generate (voicedesign model) ───────────────────────────────────────────
    if args.model == "voicedesign":
        if args.voice:
            voices = {"custom": args.voice}
        elif args.emotion:
            voices = {e: EMOTION_PRESETS[e] for e in args.emotion}
        else:
            voices = dict(EMOTION_PRESETS)

        if args.dry_run:
            print("Dry run — would generate (voicedesign):")
            for emotion_name in voices:
                out = output_dir / f"{slug}_{emotion_name}.wav"
                print(f"  {emotion_name}: {out.relative_to(ROOT)}")
            return

        client = build_client()
        for emotion_name, voice_desc in voices.items():
            out_path = output_dir / f"{slug}_{emotion_name}.wav"
            print(f"Generating [{emotion_name}]...")
            try:
                result = generate_audio_voicedesign(client, overview_text, voice_desc, out_path)
                size_kb = result.stat().st_size / 1024
                print(f"  ✓ {result.relative_to(ROOT)}  ({size_kb:.0f} KB)")
            except Exception as e:
                print(f"  ✗ Failed: {e}", file=sys.stderr)

    # ── Generate (tts model with built-in voices) ─────────────────────────────
    else:
        preset = AUDIO_TAG_PRESETS[args.tag_preset]

        if args.dry_run:
            out = output_dir / f"{slug}_{args.voice_id}_{args.tag_preset}.wav"
            print(f"Dry run — would generate (tts):")
            print(f"  Voice:  {args.voice_id}")
            print(f"  Style:  {preset['style']}")
            print(f"  Output: {out.relative_to(ROOT)}")
            return

        client = build_client()
        out_path = output_dir / f"{slug}_{args.voice_id}_{args.tag_preset}.wav"
        print(f"Generating [{args.voice_id} / {args.tag_preset}]...")
        try:
            result = generate_audio_builtin(
                client,
                overview_text,
                style_tag=preset["style"],
                voice_id=args.voice_id,
                output_path=out_path,
            )
            size_kb = result.stat().st_size / 1024
            print(f"  ✓ {result.relative_to(ROOT)}  ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
