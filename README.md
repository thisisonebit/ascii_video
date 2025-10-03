# ASCII/Block Video Player

Play videos (files or your camera) directly in your terminal as ASCII art or high‑density colored half‑blocks. Smooth rendering, robust timing, and flexible sizing.

## Features

- Smooth terminal rendering with ANSI cursor controls (no flicker clears)
- Robust timing with drift correction and optional frame skipping
- Flexible sizing: fixed width or fit to current terminal
- Multiple rendering modes:
	- Plain ASCII (grayscale)
	- Colorized ASCII (`--color`)
	- High‑density colored half‑blocks (`--blocks`, uses the ▀ glyph with truecolor FG/BG)
- Extras: invert brightness mapping, loop playback, camera source, alternate screen buffer

## Requirements

- Python 3.9+
- Terminal with UTF‑8 and preferably TrueColor (24‑bit color) support
	- iTerm2, Apple Terminal.app, most modern terminals are fine.
- macOS, Linux, or Windows (WSL/ConPTY terminals supporting ANSI)

Python dependencies:

- opencv-python
- numpy

Install them via the provided `requirements.txt` (see below).

Optional tools (useful on macOS to list camera devices):

- ffmpeg (to list AVFoundation devices)

## Installation

Create and activate a virtual environment (recommended), then install dependencies.

```bash
# macOS / zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer not to use a venv:

```bash
pip3 install --user -r requirements.txt
```

## Quick start

This script defaults to an interactive setup flow.

```bash
python index.py
```

Follow the prompts to choose: video file vs. camera, width or fit, FPS override, ASCII vs blocks, color, etc.

### Camera quick start (interactive)

1) Start: `python index.py`
2) Answer “yes” to “Use camera as source?”
3) Enter camera index (usually `0`)
4) Choose ASCII/plain or blocks/color per prompts

### Non‑interactive CLI mode (advanced)

The file defines a full CLI (`main()`), but `__main__` currently launches the interactive flow. To run with flags without editing the file, you can invoke the CLI entry programmatically:

```bash
python -c 'import sys, index; index.main(sys.argv[1:])' -- \
	--camera 0 --width 80 --fit --fps 24 --invert
```

If you prefer to always use flags, you can change the last line of `index.py` from `interactive_main()` to `main()`.

## Usage (CLI flags)

Flags are available when calling `main()` (see “Non‑interactive CLI mode”).

```
positional arguments:
	source                Video file path (default) or camera index when --camera is set.

options:
	-w, --width INT       Target terminal width in columns. If omitted with --fit, uses terminal width.
	--fit                 Fit rendering to the current terminal size (columns and rows).
	--fps FLOAT           Override target FPS. Otherwise uses video FPS, falling back to 30.
	--charset STRING      Characters from light to dark for ASCII mode. Default: " .:-=+*#%@"
	--invert              Invert brightness mapping (light <-> dark).
	--color               Colorize ASCII characters using per‑cell color (slower).
	--blocks              High‑density colored half‑blocks (▀) with truecolor FG/BG. Implies color output.
	--frame-skip          Skip frames if playback falls behind schedule.
	--loop                Loop playback when reaching the end of the source.
	--alt-screen          Use the terminal's alternate screen (restores original on exit).
	--camera              Interpret SOURCE as a camera index (e.g., 0) instead of a file path.
```

Examples:

```bash
# Play a video file at 80 cols, ASCII
python -c 'import sys, index; index.main(sys.argv[1:])' -- myvideo.mp4 -w 80

# Camera 0, fit to terminal, high‑density colored blocks
python -c 'import sys, index; index.main(sys.argv[1:])' -- --camera 0 --fit --blocks

# File, colorized ASCII, invert, 24 FPS, allow frame skipping
python -c 'import sys, index; index.main(sys.argv[1:])' -- myvideo.mp4 --color --invert --fps 24 --frame-skip
```

## Modes and quality

- ASCII (plain): fastest, grayscale characters only.
- Colorized ASCII (`--color`): per‑cell foreground color; a bit slower.
- Half‑blocks (`--blocks`): best visual density and color fidelity; each cell shows two vertical pixels using a single “▀” glyph with different FG/BG truecolor.

Tip: If your terminal supports TrueColor, `--blocks` usually looks best. If performance is tight, try plain ASCII without `--color`.

## Sizing

- `--width N`: set a fixed number of columns; rows are computed from the input aspect and an internal font aspect (≈ 0.5 height/width).
- `--fit`: scale to fit both terminal width and height while preserving aspect.

The renderer accounts for typical terminal cell aspect ratio when estimating rows so circles don’t look squashed.

## Timing and smoothness

- FPS is chosen by `--fps` override, else the file’s FPS, else 30.
- Drift correction keeps long sessions stable.
- `--frame-skip` lets the player drop frames to catch up if your machine falls behind.
- `--loop` restarts files when they end; for cameras, reading simply continues.

## Camera index

When `--camera` is set, `source` must be an integer, e.g., `0` for the default webcam, `1` for the next device, etc. The program uses `cv2.VideoCapture(index)`.

macOS: you can list AVFoundation devices with ffmpeg (optional):

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

Permissions: if the camera is black or fails to open, grant your terminal app access in Settings → Privacy & Security → Camera.

## Charset and invert

- `--charset`: set characters from light → dark. Default is `" .:-=+*#%@"`.
- `--invert`: flips brightness mapping (helpful for dark terminals or striking looks). In `--blocks` mode, invert flips RGB colors (255 - color).

## Terminal compatibility

- TrueColor is recommended for `--color` and especially `--blocks`.
- If colors look dull or wrong, ensure your terminal is configured for 24‑bit color and not downgrading to 256 colors.
- UTF‑8 must be enabled so the “▀” glyph renders correctly.

## Performance tips

- Prefer `--blocks` for best visual density (can be heavier on CPU than plain ASCII but typically efficient).
- Reduce `--width` or avoid `--fit` on very large terminals if CPU is high.
- Avoid `--color` in ASCII mode for maximum speed.
- Use `--fps` to cap the rate, and `--frame-skip` to keep audio‑like smoothness (for files) under load.

## How it works (brief)

- Frames are read with OpenCV (BGR).
- In ASCII mode: frames are resized, converted to gray, mapped to characters (and optional per‑cell color).
- In block mode: frames are resized to double row count, split into top/bottom rows per printed line, then each printed character is a “▀” with FG=top color and BG=bottom color.
- Rendering uses ANSI: move cursor home, print the frame, reset colors; optional alternate screen keeps your main buffer clean.

## Troubleshooting

- “Failed to open source”: path is wrong, camera index invalid, or permissions missing.
- Camera shows black image: grant Terminal/iTerm2 camera permission in macOS settings.
- Colors look off: ensure TrueColor is enabled; try a different terminal.
- Garbled or wrapped lines: set an appropriate `--width` or use `--fit`.
- High CPU: reduce `--width`, disable `--color`, or enable `--frame-skip` and/or lower `--fps`.

## License

MIT — see `LICENSE`.

## Acknowledgements

Built with OpenCV and NumPy. TrueColor ANSI sequences per common terminal conventions (`38;2;R;G;B` and `48;2;R;G;B`).

