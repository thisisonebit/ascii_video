#!/usr/bin/env python3
"""
ASCII/Block Video Player - CLI

Features:
- Smooth terminal rendering using ANSI cursor controls (no flicker clears)
- Robust timing with drift correction and optional frame skipping
- Flexible sizing: fixed width or fit to terminal
- Multiple rendering modes:)
  - Colored ASCII (--color)
  - High-density colored half-blocks (--blocks)
- Extras: invert brightness mapping, loop playback, camera source

Dependencies:
  pip install opencv-python numpy
"""

import argparse
import os
import sys
import time
import math
import shutil
import signal
from typing import List, Tuple, Optional

try:
    import cv2
    import numpy as np
except Exception as e:
    sys.stderr.write("Error: This script requires 'opencv-python' and 'numpy'.\n")
    sys.stderr.write("Install them with: pip install opencv-python numpy\n")
    raise


DEFAULT_CHARS = " .:-=+*#%@"
CSI = "\x1b["

#used to get the terminal size
def get_term_size() -> Tuple[int, int]:
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines
    except Exception:
        return (80, 24)

#ok so this is basically a clamp function to keep the values between a specific range so as to not go out of bounds and either mess up openCV or the terminal rendering
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

#so this class basically makes it possible to render stuff. It uses ANSI escape codes to control the terminal.
class TerminalRenderer:
    
    #initializes the terminal renderer. 
    def __init__(self, use_alt_screen: bool = False):
        self.use_alt_screen = use_alt_screen
        self._active = False

    def __enter__(self):
        # Hide cursor
        sys.stdout.write(f"{CSI}?25l")
        # Enter alternate screen buffer so as to not mess up the original terminal content
        if self.use_alt_screen:
            sys.stdout.write(f"{CSI}?1049h")
        # Move cursor home and clear screen initially
        sys.stdout.write(f"{CSI}H{CSI}2J")
        sys.stdout.flush()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        # Reset colors
        sys.stdout.write(f"{CSI}0m")
        # Show cursor
        sys.stdout.write(f"{CSI}?25h")
        # Leave alternate screen if used
        if self.use_alt_screen:
            sys.stdout.write(f"{CSI}?1049l")
        sys.stdout.flush()
        self._active = False

    def render(self, lines: List[str]):
        # Move cursor to home
        sys.stdout.write(f"{CSI}H")
        # Write all lines joined once for efficiency
        sys.stdout.write("\n".join(lines))
        # Reset color at end of frame to avoid bleed
        sys.stdout.write(f"{CSI}0m")
        sys.stdout.flush()


class FrameConverter:
    """
    Converts raw frames (BGR) to terminal-friendly strings.
    Provides ASCII and colored half-block modes.
    """

    def __init__(
        self,
        charset: str = DEFAULT_CHARS,
        invert: bool = False,
        use_color_ascii: bool = False,
        use_blocks: bool = False,
        font_aspect: float = 0.5,
    ):
        """
        font_aspect: approximate character cell height/width ratio.
                     Typical terminals ~2.0 height:width => aspect ~0.5
        """
        self.charset = charset if charset else DEFAULT_CHARS
        self.invert = invert
        self.use_color_ascii = use_color_ascii
        self.use_blocks = use_blocks
        self.font_aspect = font_aspect

    def _compute_target_size_ascii(
        self,
        frame_w: int,
        frame_h: int,
        target_cols: Optional[int],
        fit_terminal: bool,
    ) -> Tuple[int, int]:
        """
        Compute target width (columns) and rows for ASCII mode, respecting terminal size if fit_terminal.
        """
        term_cols, term_rows = get_term_size()

        if target_cols is None:
            # Default to terminal width when fitting, else 80 columns
            target_cols = term_cols if fit_terminal else 80

        #calculate the rows based on the font aspect ratio and the video frame dimensions
        rows = max(1, int(frame_h * (target_cols / frame_w) * self.font_aspect))

        if fit_terminal:
            # Clamp to terminal bounds by uniformly scaling columns
            if rows > term_rows:
                scale = term_rows / rows
                target_cols = max(1, int(target_cols * scale))
                rows = max(1, int(rows * scale))

            target_cols = clamp(target_cols, 1, term_cols)

        return target_cols, rows

    def _compute_target_size_blocks(
        self,
        frame_w: int,
        frame_h: int,
        target_cols: Optional[int],
        fit_terminal: bool,
    ) -> Tuple[int, int, int]:
        """
        For half-block mode:
        - Compute target_cols (columns),
        - printed_rows (terminal rows actually printed),
        - resize_rows (image rows, equals printed_rows * 2).
        """
        term_cols, term_rows = get_term_size()

        if target_cols is None:
            target_cols = term_cols if fit_terminal else 80

        # Compute how many printed rows we want using the font aspect
        # Each printed row represents 2 image rows.
        printed_rows = max(1, int(frame_h * (target_cols / frame_w) * self.font_aspect))

        if fit_terminal:
            if printed_rows > term_rows:
                scale = term_rows / printed_rows
                target_cols = max(1, int(target_cols * scale))
                printed_rows = max(1, int(printed_rows * scale))
            target_cols = clamp(target_cols, 1, term_cols)
            printed_rows = clamp(printed_rows, 1, term_rows)

        resize_rows = max(2, printed_rows * 2)
        return target_cols, printed_rows, resize_rows

    @staticmethod
    def _bgr_to_gray(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _map_char(self, norm: float) -> str:
        # norm in [0,1], map to charset
        if self.invert:
            norm = 1.0 - norm
        idx = int(norm * (len(self.charset) - 1))
        return self.charset[idx]

    def convert_ascii(
        self,
        frame_bgr: np.ndarray,
        target_cols: Optional[int] = None,
        fit_terminal: bool = False,
    ) -> List[str]:
        """
        Convert frame to ASCII lines (optionally colored).
        """
        h, w = frame_bgr.shape[:2]
        cols, rows = self._compute_target_size_ascii(w, h, target_cols, fit_terminal)

        if cols <= 0 or rows <= 0:
            return [""]

        # Resize for sampling
        resized = cv2.resize(frame_bgr, (cols, rows), interpolation=cv2.INTER_AREA)

        if self.use_color_ascii:
            # Use colorized ASCII by coloring foreground per cell
            rgb = self._bgr_to_rgb(resized)
            gray = self._bgr_to_gray(resized).astype(np.float32) / 255.0
            lines: List[str] = []
            for y in range(rows):
                row_chars: List[str] = []
                for x in range(cols):
                    ch = self._map_char(gray[y, x])
                    r, g, b = rgb[y, x]
                    row_chars.append(f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m{ch}")
                lines.append("".join(row_chars) + "\x1b[0m")
            return lines
        else:
            # Plain grayscale ASCII
            gray = self._bgr_to_gray(resized).astype(np.float32) / 255.0
            if self.invert:
                gray = 1.0 - gray
            # Vectorized to speed up, then map to chars
            idx = np.clip(
                (gray * (len(self.charset) - 1)).astype(np.int32),
                0,
                len(self.charset) - 1,
            )
            char_array = np.array(list(self.charset), dtype="<U1")
            lines = ["".join(char_array[row]) for row in idx]
            return lines

    def convert_blocks(
        self,
        frame_bgr: np.ndarray,
        target_cols: Optional[int] = None,
        fit_terminal: bool = False,
    ) -> List[str]:
        """
        Convert frame to high-density colored half-blocks.
        Each terminal cell renders two vertical pixels using foreground (top) and background (bottom) colors.
        """
        h, w = frame_bgr.shape[:2]
        cols, printed_rows, resize_rows = self._compute_target_size_blocks(
            w, h, target_cols, fit_terminal
        )

        if cols <= 0 or printed_rows <= 0:
            return [""]

        # Resize to (cols, resize_rows)
        resized = cv2.resize(
            frame_bgr, (cols, resize_rows), interpolation=cv2.INTER_AREA
        )
        rgb = self._bgr_to_rgb(resized)

        # If invert is requested, we'll invert brightness by flipping colors toward inverse.
        # For colored blocks, invert by mapping RGB -> 255 - RGB
        if self.invert:
            rgb = 255 - rgb

        lines: List[str] = []
        # Each printed row uses two image rows: y*2 (top) and y*2+1 (bottom)
        for y in range(0, resize_rows, 2):
            if y + 1 >= resize_rows:
                # If odd number of rows, pair bottom with black
                top = rgb[y, :, :]
                bottom = np.zeros_like(top)
            else:
                top = rgb[y, :, :]
                bottom = rgb[y + 1, :, :]

            # Build a line using '▀' with fg=top, bg=bottom
            # 38;2;r;g;b = foreground, 48;2;r;g;b = background
            parts: List[str] = []
            for x in range(cols):
                r1, g1, b1 = top[x]
                r2, g2, b2 = bottom[x]
                parts.append(
                    f"\x1b[38;2;{int(r1)};{int(g1)};{int(b1)}m"
                    f"\x1b[48;2;{int(r2)};{int(g2)};{int(b2)}m▀"
                )
            # Reset at end of line to avoid color bleed on shorter next lines
            parts.append("\x1b[0m")
            lines.append("".join(parts))

        # Ensure we do not exceed printed_rows (due to rounding)
        if len(lines) > printed_rows:
            lines = lines[:printed_rows]

        return lines

#capture the video from either a file or a camera
def open_capture(source: str, camera: bool = False) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture from either a file path or a camera index.
    """
    if camera:
        try:
            index = int(source)
        except ValueError:
            raise ValueError(
                "When --camera is specified, SOURCE must be an integer camera index (e.g., 0)."
            )
        cap = cv2.VideoCapture(index)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    return cap

#compute the target fps, pretty much
def compute_target_fps(cap: cv2.VideoCapture, fps_override: float) -> float:
    """
    Decide the target FPS. Prefer override, then video fps, else 30.
    """
    if fps_override and fps_override > 0:
        return float(fps_override)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps is None or fps <= 0 or math.isnan(fps):
        return 30.0
    return float(fps)

#ctrl or cmd c exit handler
def install_sigint_handler():
    #Make Ctrl-C responsive by avoiding delayed KeyboardInterrupt during sleep on some platforms.
    signal.signal(signal.SIGINT, signal.default_int_handler)


def play(
    source: str,
    width: Optional[int],
    fit: bool,
    fps_override: Optional[float],
    charset: str,
    invert: bool,
    color_ascii: bool,
    blocks: bool,
    frame_skip: bool,
    loop: bool,
    alt_screen: bool,
    camera: bool,
):
    """
    Main playback loop with precise timing and rendering.
    """
    converter = FrameConverter(
        charset=charset,
        invert=invert,
        use_color_ascii=color_ascii,
        use_blocks=blocks,
        font_aspect=0.5,
    )

    cap = open_capture(source, camera=camera)
    target_fps = compute_target_fps(cap, fps_override or 0.0)
    frame_period = 1.0 / target_fps

    # Prepare terminal
    install_sigint_handler()
    with TerminalRenderer(use_alt_screen=alt_screen) as renderer:
        try:
            next_frame_time = time.perf_counter()
            last_render_dimensions: Tuple[int, int] = (0, 0)  # cols, rows printed

            while True:
                ret, frame = cap.read()
                if not ret:
                    if loop:
                        # Restart video
                        if not camera:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            # For cameras, just continue to next read attempt
                            continue
                    break

                # Choose conversion mode
                if blocks:
                    lines = converter.convert_blocks(
                        frame, target_cols=width, fit_terminal=fit
                    )
                else:
                    lines = converter.convert_ascii(
                        frame, target_cols=width, fit_terminal=fit
                    )

                # If the new frame has fewer lines than previously drawn,
                # clear the remainder to avoid leftover rows on resize.
                cols_now = len(lines[0]) if lines else 0
                rows_now = len(lines)
                cols_prev, rows_prev = last_render_dimensions
                if rows_prev > rows_now:
                    pad = [" " * (cols_now or cols_prev)] * (rows_prev - rows_now)
                    lines = lines + pad

                renderer.render(lines)
                last_render_dimensions = (cols_now, rows_now)

                # Timing with drift correction
                now = time.perf_counter()
                if next_frame_time <= now:
                    # We're due (or late). Schedule next frame after period from now to reduce cumulative drift.
                    next_frame_time = now + frame_period
                else:
                    # We're early; sleep the remaining time if it's meaningful
                    sleep_time = next_frame_time - now
                    if sleep_time > 0.0005:
                        time.sleep(sleep_time)
                        next_frame_time += frame_period
                    else:
                        next_frame_time += frame_period

                # If we fell behind substantially and frame skipping is enabled, drop frames to catch up
                if frame_skip:
                    behind = time.perf_counter() - next_frame_time
                    # For very small delays, don't skip to avoid jitter
                    if behind > frame_period * 1.5:
                        # Skip enough frames to catch up approximately
                        skip_count = int(behind // frame_period)
                        for _ in range(skip_count):
                            ret, _ = cap.read()
                            if not ret:
                                break
                        # Re-anchor next_frame_time to "now + period" to avoid spiraling
                        next_frame_time = time.perf_counter() + frame_period

        except KeyboardInterrupt:
            # Graceful exit on Ctrl-C
            pass
        finally:
            cap.release()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Play a video as ASCII or colored half-blocks directly in your terminal."
    )
    p.add_argument(
        "source",
        help="Video file path (default) or camera index when --camera is set.",
    )
    p.add_argument(
        "-w",
        "--width",
        type=int,
        default=None,
        help="Target terminal width in columns. If omitted with --fit, uses terminal width.",
    )
    p.add_argument(
        "--fit",
        action="store_true",
        help="Fit rendering to the current terminal size (columns and rows).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override target FPS. Otherwise uses video FPS, falling back to 30.",
    )
    p.add_argument(
        "--charset",
        type=str,
        default=DEFAULT_CHARS,
        help=f"Characters from light to dark for ASCII mode. Default: '{DEFAULT_CHARS}'",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert brightness mapping (light <-> dark).",
    )
    p.add_argument(
        "--color",
        action="store_true",
        help="Colorize ASCII characters using per-cell color (slower).",
    )
    p.add_argument(
        "--blocks",
        action="store_true",
        help="Use high-density colored half-blocks (▀) with truecolor foreground/background.",
    )
    p.add_argument(
        "--frame-skip",
        action="store_true",
        help="Skip frames if playback falls behind schedule.",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback when reaching the end of the source.",
    )
    p.add_argument(
        "--alt-screen",
        action="store_true",
        help="Render in the terminal's alternate screen buffer (restores original screen on exit).",
    )
    p.add_argument(
        "--camera",
        action="store_true",
        help="Interpret SOURCE as a camera index (e.g., 0) instead of a file path.",
    )
    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate arguments
    if args.camera:
        # Ensure source is an int-like string
        try:
            int(args.source)
        except ValueError:
            parser.error(
                "When using --camera, SOURCE must be an integer camera index (e.g., 0)."
            )
    else:
        # Check file existence here for early feedback
        if not os.path.exists(args.source):
            parser.error(f"File not found: {args.source}")

    # Warn if incompatible flags chosen
    if args.blocks and args.color:
        # Blocks mode is inherently colorized; --color is redundant
        sys.stderr.write("Note: --blocks implies colored output; --color is ignored.\n")

    try:
        play(
            source=args.source,
            width=args.width,
            fit=args.fit,
            fps_override=args.fps,
            charset=args.charset,
            invert=args.invert,
            color_ascii=(args.color and not args.blocks),
            blocks=args.blocks,
            frame_skip=args.frame_skip,
            loop=args.loop,
            alt_screen=args.alt_screen,
            camera=args.camera,
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

#just the menu driven program
def interactive_main():
    def ask_bool(prompt: str, default: bool = False) -> bool:
        suffix = " [Y/n]: " if default else " [y/N]: "
        while True:
            ans = input(prompt + suffix).strip().lower()
            if ans == "":
                return default
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            print("Please enter y or n.")

    def ask_int(prompt: str, default=None):
        while True:
            s = input(prompt).strip()
            if s == "":
                return default
            try:
                return int(s)
            except ValueError:
                print("Please enter a valid integer.")

    def ask_float(prompt: str, default=None):
        while True:
            s = input(prompt).strip()
            if s == "":
                return default
            try:
                return float(s)
            except ValueError:
                print("Please enter a valid number (e.g., 24 or 29.97).")

    print("Terminal Video Player (ASCII/Blocks) - Interactive Setup")

    # --- Basic options (shown to every user) ---
    camera = ask_bool("Use camera as source?", default=False)
    if camera:
        source = input("Enter camera index (e.g., 0): ").strip()
        try:
            int(source)
        except ValueError:
            print("Invalid camera index; defaulting to 0.")
            source = "0"
    else:
        while True:
            source = input("Enter path to video file: ").strip()
            if os.path.exists(source):
                break
            print("File not found. Please enter a valid path.")

    width = ask_int(
        "Enter terminal width in columns (leave blank to auto): ", default=None
    )
    fit = ask_bool("Fit rendering to terminal size?", default=False)

    # Mode selection is part of the basic flow so users can start quickly
    blocks = ask_bool(
        "Use high-density colored half-blocks (recommended for best quality)?",
        default=False,
    )
    color_ascii = False
    if not blocks:
        color_ascii = ask_bool("Colorize ASCII characters?", default=False)

    # --- Advanced options (hidden unless requested) ---
    advanced = ask_bool("Configure advanced options?", default=False)

    if advanced:
        fps_override = ask_float(
            "Override FPS (leave blank to use video/default): ", default=None
        )

        charset_in = input(
            f"Enter ASCII charset (light->dark) [default: {DEFAULT_CHARS}]: "
        ).strip()
        charset = charset_in if charset_in else DEFAULT_CHARS

        invert = ask_bool("Invert brightness mapping?", default=False)

        frame_skip = ask_bool("Enable frame skipping when behind?", default=True)
        loop = ask_bool("Loop playback when reaching the end?", default=False)
        alt_screen = ask_bool(
            "Use alternate screen buffer (cleaner experience)?", default=True
        )
    else:
        # Use sensible defaults when advanced options are skipped
        fps_override = None
        charset = DEFAULT_CHARS
        invert = False
        frame_skip = True
        loop = False
        alt_screen = True

    try:
        play(
            source=source,
            width=width,
            fit=fit,
            fps_override=fps_override,
            charset=charset,
            invert=invert,
            color_ascii=(color_ascii and not blocks),
            blocks=blocks,
            frame_skip=frame_skip,
            loop=loop,
            alt_screen=alt_screen,
            camera=camera,
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    interactive_main()
