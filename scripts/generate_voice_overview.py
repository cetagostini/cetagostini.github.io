#!/usr/bin/env python3
"""Generate voice audio from a text file using MiMo V2.5 TTS.

Two models are available:

  mimo-v2.5-tts             — Built-in voices (Chloe, Mia, Milo, etc.) with audio tag control.
                               Embed [crying], [pause], [sigh], etc. directly in the text.
  mimo-v2.5-tts-voicedesign — Generate any voice from a text description (director mode).
                               The --voice flag describes the voice character.

Model docs: https://mimo.mi.com/models/en-US/mimo-v2.5-tts
            https://mimo.mi.com/models/en-US/mimo-v2.5-tts-voicedesign

Input is a plain .txt file — the exact text to be spoken. Audio tags like [pause],
[crying], [sigh] are supported in the text when using the tts model. Output is a
.wav file placed next to the input file.

Requires:
    pip install openai
    export MIMO_API_KEY=...   # or set in .env

Usage:
    # TTS model with built-in voice (default)
    python scripts/generate_voice_overview.py articles/my_article/audio/overview.txt

    # Specific voice and style
    python scripts/generate_voice_overview.py articles/my_article/audio/overview.txt \\
        --voice-id Milo --style "Serious Magnetic"

    # Voicedesign model with custom voice description
    python scripts/generate_voice_overview.py articles/my_article/audio/overview.txt \\
        --model voicedesign --voice "A warm, contemplative female voice"

    # List .txt files in articles/*/audio/
    python scripts/generate_voice_overview.py --list
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import textwrap
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = ROOT / "articles"

# ── Built-in voices ───────────────────────────────────────────────────────────

BUILTIN_VOICES = ["mimo_default", "Mia", "Chloe", "Milo", "Dean"]

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

    base_url = os.environ.get("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_audio_tts(
    client: OpenAI,
    text: str,
    voice_id: str,
    output_path: Path,
    style: str | None = None,
) -> Path:
    """Generate audio using mimo-v2.5-tts (built-in voices + audio tags).

    Args:
        client: OpenAI-compatible client.
        text: Text to synthesize. May contain [audio tags] like [pause], [crying], etc.
        voice_id: Built-in voice ID (Chloe, Mia, Milo, Dean, mimo_default).
        output_path: Where to write the .wav.
        style: Optional style prefix, e.g. "Gentle Warm". Prepended as (style) tag.
    """
    messages: list[dict[str, str]] = []
    content = f"({style}){text}" if style else text
    messages.append({"role": "assistant", "content": content})

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


def generate_audio_voicedesign(
    client: OpenAI,
    text: str,
    voice_description: str,
    output_path: Path,
) -> Path:
    """Generate audio using mimo-v2.5-tts-voicedesign (voice from text description).

    Args:
        client: OpenAI-compatible client.
        text: Text to synthesize.
        voice_description: Natural-language voice description (director mode).
        output_path: Where to write the .wav.
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

def find_text_files() -> list[Path]:
    """Find all .txt files in articles/*/audio/."""
    files: list[Path] = []
    if ARTICLES_DIR.exists():
        for article_dir in sorted(ARTICLES_DIR.iterdir()):
            audio_dir = article_dir / "audio"
            if audio_dir.is_dir():
                files.extend(sorted(audio_dir.glob("*.txt")))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate voice audio from a text file using MiMo V2.5 TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            models:
              tts (default)       Built-in voices with audio tag control.
                                  Text can contain [pause], [crying], [sigh], [laugh], etc.
              voicedesign         Generate any voice from a --voice description.
                                  Plain text only (no audio tags).

            examples:
              %(prog)s articles/my_article/audio/overview.txt
              %(prog)s articles/my_article/audio/overview.txt --voice-id Milo
              %(prog)s articles/my_article/audio/overview.txt --voice-id Chloe --style "Serious Magnetic"
              %(prog)s articles/my_article/audio/overview.txt --model voicedesign --voice "A deep male voice..."
              %(prog)s --list
        """),
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to a .txt file containing the text to synthesize",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["tts", "voicedesign"],
        default="tts",
        help="TTS model. Default: tts",
    )
    parser.add_argument(
        "--voice-id",
        choices=BUILTIN_VOICES,
        default="Chloe",
        help="Built-in voice for tts model. Default: Chloe",
    )
    parser.add_argument(
        "--style",
        "-s",
        help="Style tag for tts model, e.g. 'Gentle Warm', 'Serious Magnetic'. "
             "Prepended as (style) to the text.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        help="Voice description for voicedesign model (director mode).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output .wav path. Default: same name as input with .wav extension.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List .txt files in articles/*/audio/ and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── List mode ──────────────────────────────────────────────────────────────
    if args.list:
        files = find_text_files()
        if not files:
            print("No .txt files found in articles/*/audio/")
            return
        print("Text files:\n")
        for f in files:
            rel = f.relative_to(ROOT)
            size = f.stat().st_size
            print(f"  {rel}  ({size} bytes)")
        return

    # ── Validate input ─────────────────────────────────────────────────────────
    if not args.input:
        print("Error: input .txt file is required (unless using --list)", file=sys.stderr)
        sys.exit(1)

    input_path: Path = args.input
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix != ".txt":
        print(f"Warning: Expected a .txt file, got {input_path.suffix}", file=sys.stderr)

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        print("Error: Input file is empty.", file=sys.stderr)
        sys.exit(1)

    # Output path: same location, .wav extension
    output_path = args.output if args.output else input_path.with_suffix(".wav")
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    # ── Print info ─────────────────────────────────────────────────────────────
    print(f"Input:   {input_path.relative_to(ROOT)}")
    print(f"Output:  {output_path.relative_to(ROOT)}")
    print(f"Model:   mimo-v2.5-tts{'-voicedesign' if args.model == 'voicedesign' else ''}")
    print(f"Text:    {len(text)} chars (~{len(text) // 6} words)")
    if args.model == "tts":
        print(f"Voice:   {args.voice_id}")
        if args.style:
            print(f"Style:   ({args.style})")
    else:
        print(f"Voice:   {args.voice or '(default)'}")
    print()

    if args.dry_run:
        print("Dry run — no API call made.")
        return

    # ── Generate ───────────────────────────────────────────────────────────────
    client = build_client()

    try:
        if args.model == "tts":
            result = generate_audio_tts(
                client, text,
                voice_id=args.voice_id,
                output_path=output_path,
                style=args.style,
            )
        else:
            voice_desc = args.voice or "A clear, natural voice with moderate pacing."
            result = generate_audio_voicedesign(
                client, text,
                voice_description=voice_desc,
                output_path=output_path,
            )
        size_kb = result.stat().st_size / 1024
        print(f"✓ {result.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"✗ Failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
