"""Build the bundled ACDC sequence-playback video used by the web demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from saferestore.video import build_sequence_video

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = (
    ROOT / "data" / "raw" / "ACDC_images" / "rgb_anon" / "fog" / "test" / "GOPR0475"
)
DEFAULT_OUTPUT = ROOT / "runs" / "assets" / "acdc_fog_sequence.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()

    frames = sorted(args.sequence.glob("*.png"))[: args.frames]
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {args.sequence}")
    output = build_sequence_video(frames, args.output, args.fps, args.max_side)
    print(f"Built {output} from {len(frames)} ACDC frames")


if __name__ == "__main__":
    main()
