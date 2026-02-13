#!/usr/bin/env python3
"""Motion extraction from a video using Posy's invert+opacity time-shift trick."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROCESS_MODE = "invert50"
MAX_PRE_BLUR = 49
MAX_CONTRAST = 10.0
MAX_SATURATION = 3.0
FFMPEG_PATH_ENV = "MOTION_FFMPEG_PATH"


class _AsyncVideoWritePool:
    def __init__(
        self,
        writer: cv2.VideoWriter,
        frame_shape: tuple[int, int, int],
        pool_size: int = 4,
    ) -> None:
        if pool_size < 2:
            raise ValueError("pool_size must be >= 2")
        self.writer = writer
        self.buffers = [np.empty(frame_shape, dtype=np.uint8) for _ in range(pool_size)]
        self.free_indices: queue.Queue[int] = queue.Queue()
        self.ready_indices: queue.Queue[int | None] = queue.Queue()
        for i in range(pool_size):
            self.free_indices.put(i)

        self.worker_error: Exception | None = None
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self) -> None:
        try:
            while True:
                index = self.ready_indices.get()
                if index is None:
                    break
                self.writer.write(self.buffers[index])
                self.free_indices.put(index)
        except Exception as exc:  # pragma: no cover - protective fallback
            self.worker_error = exc

    def acquire(self) -> tuple[int, np.ndarray]:
        while True:
            if self.worker_error is not None:
                raise RuntimeError("Async writer worker failed") from self.worker_error
            try:
                index = self.free_indices.get(timeout=0.1)
                return index, self.buffers[index]
            except queue.Empty:
                continue

    def submit(self, index: int) -> None:
        if self.worker_error is not None:
            raise RuntimeError("Async writer worker failed") from self.worker_error
        self.ready_indices.put(index)

    def close(self) -> None:
        self.ready_indices.put(None)
        self.worker.join()
        if self.worker_error is not None:
            raise RuntimeError("Async writer worker failed") from self.worker_error


def _to_gray3(
    frame_bgr: np.ndarray,
    gray_buffer: np.ndarray | None = None,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    if gray_buffer is None or gray_buffer.shape != frame_bgr.shape[:2]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_buffer
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=gray)

    if dst is None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR, dst=dst)
    return dst


def _build_contrast_lut(mode: str, contrast: float) -> np.ndarray | None:
    if contrast == 1.0:
        return None
    x = np.arange(256, dtype=np.float32) / 255.0
    if mode == "invert50":
        y = (x - 0.5) * contrast + 0.5
    elif mode == "difference":
        y = x * contrast
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def _build_overlay_highlight_lut(mode: str) -> np.ndarray:
    x = np.arange(256, dtype=np.int16)
    if mode == "invert50":
        # Neutral (128) maps to 0; max motion maps near 255.
        y = np.minimum(255, np.abs(x - 128) * 2)
    elif mode == "difference":
        y = x
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return y.astype(np.uint8)


def _to_uint8_bgr(motion: np.ndarray, grayscale: bool) -> np.ndarray:
    if grayscale:
        return _to_gray3(motion)
    return motion


def _apply_saturation_bgr_in_place(
    frame: np.ndarray,
    saturation: float,
    hsv_buffer: np.ndarray,
    saturation_lut: np.ndarray,
) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    if saturation == 1.0:
        return
    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=hsv_buffer)
    hsv_buffer[:, :, 1] = saturation_lut[hsv_buffer[:, :, 1]]
    cv2.cvtColor(hsv_buffer, cv2.COLOR_HSV2BGR, dst=frame)


def _accumulate_trails_in_place(
    frame: np.ndarray,
    accumulation_buffer: np.ndarray,
    output_float_buffer: np.ndarray,
    decay: float,
    mode: str,
) -> None:
    if mode == "invert50":
        neutral = 128.0
    elif mode == "difference":
        neutral = 0.0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    np.copyto(output_float_buffer, frame, casting="unsafe")
    if neutral != 0.0:
        output_float_buffer -= neutral
    np.multiply(accumulation_buffer, decay, out=accumulation_buffer)
    np.add(accumulation_buffer, output_float_buffer, out=accumulation_buffer)
    np.copyto(output_float_buffer, accumulation_buffer)
    if neutral != 0.0:
        output_float_buffer += neutral
    np.clip(output_float_buffer, 0.0, 255.0, out=output_float_buffer)
    np.copyto(frame, output_float_buffer, casting="unsafe")


def overlay_motion_on_original(
    original: np.ndarray,
    motion_visualization: np.ndarray,
    mode: str,
    overlay_strength: float = 1.0,
    highlight_lut: np.ndarray | None = None,
    highlight_buffer: np.ndarray | None = None,
    orig_u16_buffer: np.ndarray | None = None,
    high_u16_buffer: np.ndarray | None = None,
    inv_prod_u16_buffer: np.ndarray | None = None,
    out_u8_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay motion visualization on the original frame.

    For invert50, neutral gray is treated as no-motion. For difference mode,
    black is treated as no-motion. Overlay uses a screen blend for highlights.
    """
    if original.shape != motion_visualization.shape:
        raise ValueError("original and motion_visualization must have the same shape")
    if not 0.0 <= overlay_strength <= 1.0:
        raise ValueError("overlay_strength must be in [0, 1]")

    if highlight_lut is None:
        highlight_lut = _build_overlay_highlight_lut(mode)
    if highlight_buffer is not None and highlight_buffer.shape == original.shape:
        highlight = highlight_buffer
        cv2.LUT(motion_visualization, highlight_lut, dst=highlight)
    else:
        highlight = cv2.LUT(motion_visualization, highlight_lut)

    if overlay_strength == 0.0:
        highlight.fill(0)
    elif overlay_strength != 1.0:
        cv2.convertScaleAbs(highlight, alpha=overlay_strength, dst=highlight)

    can_use_buffers = (
        orig_u16_buffer is not None
        and high_u16_buffer is not None
        and inv_prod_u16_buffer is not None
        and orig_u16_buffer.shape == original.shape
        and high_u16_buffer.shape == original.shape
        and inv_prod_u16_buffer.shape == original.shape
        and orig_u16_buffer.dtype == np.uint16
        and high_u16_buffer.dtype == np.uint16
        and inv_prod_u16_buffer.dtype == np.uint16
    )

    if can_use_buffers:
        # Screen blend in integer space:
        # out = 255 - ((255-orig)*(255-highlight))/255
        np.copyto(orig_u16_buffer, original, casting="unsafe")
        np.copyto(high_u16_buffer, highlight, casting="unsafe")
        np.subtract(255, orig_u16_buffer, out=orig_u16_buffer)
        np.subtract(255, high_u16_buffer, out=high_u16_buffer)
        np.multiply(orig_u16_buffer, high_u16_buffer, out=inv_prod_u16_buffer)
        np.add(inv_prod_u16_buffer, 127, out=inv_prod_u16_buffer, casting="unsafe")
        np.floor_divide(inv_prod_u16_buffer, 255, out=inv_prod_u16_buffer)
        np.subtract(255, inv_prod_u16_buffer, out=inv_prod_u16_buffer)

        if (
            out_u8_buffer is not None
            and out_u8_buffer.shape == original.shape
            and out_u8_buffer.dtype == np.uint8
        ):
            np.copyto(out_u8_buffer, inv_prod_u16_buffer, casting="unsafe")
            return out_u8_buffer
        return inv_prod_u16_buffer.astype(np.uint8)

    orig_u16 = original.astype(np.uint16)
    high_u16 = highlight.astype(np.uint16)
    inv_prod = (255 - orig_u16) * (255 - high_u16)
    out_u16 = 255 - ((inv_prod + 127) // 255)
    return out_u16.astype(np.uint8)


def compute_motion_frame(
    current: np.ndarray,
    reference: np.ndarray,
    mode: str = "invert50",
    contrast: float = 1.0,
    grayscale: bool = False,
    contrast_lut: np.ndarray | None = None,
    dst: np.ndarray | None = None,
    invert_buffer: np.ndarray | None = None,
    sum_u16_buffer: np.ndarray | None = None,
    gray_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one output frame from a current and reference frame."""
    if current.shape != reference.shape:
        raise ValueError("current and reference frames must have the same shape")

    if dst is not None and dst.shape == current.shape and dst.dtype == np.uint8:
        motion = dst
    else:
        motion = np.empty_like(current)

    if mode == "invert50":
        # Equivalent to round((current + (255 - reference)) / 2).
        if (
            invert_buffer is not None
            and invert_buffer.shape == reference.shape
            and invert_buffer.dtype == np.uint8
        ):
            inv = invert_buffer
            cv2.bitwise_not(reference, dst=inv)
        else:
            inv = cv2.bitwise_not(reference)

        if (
            sum_u16_buffer is not None
            and sum_u16_buffer.shape == reference.shape
            and sum_u16_buffer.dtype == np.uint16
        ):
            sum_u16 = sum_u16_buffer
            cv2.add(current, inv, dst=sum_u16, dtype=cv2.CV_16U)
        else:
            sum_u16 = cv2.add(current, inv, dtype=cv2.CV_16U)

        np.add(sum_u16, 1, out=sum_u16, casting="unsafe")
        np.right_shift(sum_u16, 1, out=sum_u16)
        np.copyto(motion, sum_u16, casting="unsafe")
    elif mode == "difference":
        cv2.absdiff(current, reference, dst=motion)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if contrast_lut is None:
        contrast_lut = _build_contrast_lut(mode, contrast)
    if contrast_lut is not None:
        cv2.LUT(motion, contrast_lut, dst=motion)

    if grayscale:
        return _to_gray3(motion, gray_buffer=gray_buffer, dst=motion)
    return _to_uint8_bgr(motion, grayscale)


def compute_motion_frame_rgb_offsets(
    current: np.ndarray,
    reference_frames_bgr: tuple[np.ndarray, np.ndarray, np.ndarray],
    mode: str = "invert50",
    contrast: float = 1.0,
    grayscale: bool = False,
    contrast_lut: np.ndarray | None = None,
    mixed_reference_buffer: np.ndarray | None = None,
    dst: np.ndarray | None = None,
    invert_buffer: np.ndarray | None = None,
    sum_u16_buffer: np.ndarray | None = None,
    gray_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one output frame with separate time offsets per B/G/R channel."""
    if len(reference_frames_bgr) != 3:
        raise ValueError("reference_frames_bgr must contain exactly 3 frames")

    for ref in reference_frames_bgr:
        if current.shape != ref.shape:
            raise ValueError("current and each reference frame must have the same shape")

    if (
        mixed_reference_buffer is not None
        and mixed_reference_buffer.shape == current.shape
        and mixed_reference_buffer.dtype == np.uint8
    ):
        mixed_reference = mixed_reference_buffer
    else:
        mixed_reference = np.empty_like(current)

    ref_b, ref_g, ref_r = reference_frames_bgr
    cv2.mixChannels([ref_b, ref_g, ref_r], [mixed_reference], [0, 0, 4, 1, 8, 2])

    return compute_motion_frame(
        current=current,
        reference=mixed_reference,
        mode=mode,
        contrast=contrast,
        grayscale=grayscale,
        contrast_lut=contrast_lut,
        dst=dst,
        invert_buffer=invert_buffer,
        sum_u16_buffer=sum_u16_buffer,
        gray_buffer=gray_buffer,
    )


def _odd_kernel(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("kernel size must be an integer") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("kernel size must be >= 0")
    if parsed > MAX_PRE_BLUR:
        raise argparse.ArgumentTypeError(f"kernel size must be <= {MAX_PRE_BLUR}")
    if parsed != 0 and parsed % 2 == 0:
        raise argparse.ArgumentTypeError("kernel size must be odd (or 0 to disable)")
    return parsed


def _non_negative_int(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _non_negative_float(value: str | float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _unit_float(value: str | float) -> float:
    parsed = _non_negative_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be <= 1.0")
    return parsed


def _contrast_float(value: str | float) -> float:
    parsed = _non_negative_float(value)
    if parsed > MAX_CONTRAST:
        raise argparse.ArgumentTypeError(f"contrast must be <= {MAX_CONTRAST:g}")
    return parsed


def _saturation_float(value: str | float) -> float:
    parsed = _non_negative_float(value)
    if parsed > MAX_SATURATION:
        raise argparse.ArgumentTypeError(f"saturation must be <= {MAX_SATURATION:g}")
    return parsed


def _rgb_offsets(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--rgb-offset-frames expects 3 comma-separated values: R,G,B"
        )
    try:
        red, green, blue = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--rgb-offset-frames values must be integers"
        ) from exc
    if red < 0 or green < 0 or blue < 0:
        raise argparse.ArgumentTypeError("--rgb-offset-frames values must be >= 0")
    return red, green, blue


def _rgb_offsets_seconds(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--rgb-offset-seconds expects 3 comma-separated values: R,G,B"
        )
    try:
        red, green, blue = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--rgb-offset-seconds values must be numbers"
        ) from exc
    if red < 0 or green < 0 or blue < 0:
        raise argparse.ArgumentTypeError("--rgb-offset-seconds values must be >= 0")
    return red, green, blue


DEFAULT_RGB_OFFSETS = (5, 10, 15)
DEFAULT_RGB_OFFSETS_TEXT = ",".join(str(v) for v in DEFAULT_RGB_OFFSETS)


def resolve_rgb_offsets_frames(
    rgb_offset_frames: tuple[int, int, int] | None,
    rgb_offset_seconds: tuple[float, float, float] | None,
    fps: float,
) -> tuple[int, int, int] | None:
    if rgb_offset_frames is not None and rgb_offset_seconds is not None:
        raise ValueError(
            "--rgb-offset-frames cannot be combined with --rgb-offset-seconds"
        )
    if rgb_offset_seconds is None:
        return rgb_offset_frames
    return tuple(int(round(v * fps)) for v in rgb_offset_seconds)


def resolve_trail_accumulation_frames(
    trail_accumulation_frames: int,
    trail_accumulation_seconds: float | None,
    fps: float,
) -> int:
    if trail_accumulation_seconds is None:
        return trail_accumulation_frames
    return int(round(trail_accumulation_seconds * fps))


def default_output_directory() -> Path:
    desktop = Path.home() / "Desktop"
    if desktop.exists() and desktop.is_dir():
        return desktop
    return Path.home()


def build_default_output_path(
    input_path: Path,
    rgb_offset_frames: tuple[int, int, int] | None,
    offset_frames: int,
    freeze_frame: int | None,
    overlay_original: bool,
) -> Path:
    base = f"{input_path.stem}_output"
    if rgb_offset_frames is not None:
        red, green, blue = rgb_offset_frames
        frame_info = f"rgb_r{red}_g{green}_b{blue}"
    elif freeze_frame is not None:
        frame_info = f"freeze{freeze_frame}"
    else:
        frame_info = f"f{offset_frames}"
    overlay_suffix = "_overlaid" if overlay_original else ""
    return default_output_directory() / f"{base}_{frame_info}{overlay_suffix}.mp4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract motion from video by blending each frame with an inverted, "
            "time-shifted duplicate."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("videos/video.mp4"),
        help="Input video path (default: videos/video.mp4).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Output video path. If omitted, it is auto-generated from input name, "
            "offset settings, and options."
        ),
    )
    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--offset-frames",
        "--frame-offset",
        dest="offset_frames",
        type=int,
        default=None,
        help="Time shift between current and reference frame (in frames).",
    )
    offset_group.add_argument(
        "--offset-seconds",
        type=float,
        default=None,
        help="Time shift in seconds. Cannot be combined with --offset-frames.",
    )
    rgb_offset_group = parser.add_mutually_exclusive_group()
    rgb_offset_group.add_argument(
        "--rgb-offset-frames",
        nargs="?",
        const=DEFAULT_RGB_OFFSETS_TEXT,
        type=_rgb_offsets,
        default=None,
        help=(
            "Per-channel time offsets in frames as R,G,B "
            "(example: --rgb-offset-frames 1,10,20). "
            "If provided without a value, defaults to 5,10,15. "
            "Cannot be combined with --rgb-offset-seconds."
        ),
    )
    rgb_offset_group.add_argument(
        "--rgb-offset-seconds",
        type=_rgb_offsets_seconds,
        default=None,
        help=(
            "Per-channel time offsets in seconds as R,G,B "
            "(example: --rgb-offset-seconds 0,0.08,0.16). "
            "Cannot be combined with --rgb-offset-frames."
        ),
    )
    parser.add_argument(
        "--freeze-frame",
        type=int,
        default=None,
        help=(
            "Use a fixed reference frame index for the whole video. "
            "Matches the 'freeze duplicate' variation."
        ),
    )
    parser.add_argument(
        "--contrast",
        type=_contrast_float,
        default=1.0,
        help=(
            "Contrast boost on the motion result (1.0 = unchanged). "
            "Range: 0 to 10."
        ),
    )
    parser.add_argument(
        "--pre-blur",
        type=_odd_kernel,
        default=0,
        help=(
            "Optional odd Gaussian blur kernel size applied before extraction. "
            "Use to suppress tiny motion/noise. Range: odd values from 1 to 49 "
            "(or 0 to disable)."
        ),
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Output a grayscale motion visualization (still encoded as 3-channel).",
    )
    parser.add_argument(
        "--overlay-original",
        action="store_true",
        help="Overlay extracted motion on top of the original video.",
    )
    parser.add_argument(
        "--overlay-strength",
        type=_unit_float,
        default=1.0,
        help=(
            "Overlay highlight strength in [0,1] when --overlay-original is used "
            "(1.0 = full strength)."
        ),
    )
    parser.add_argument(
        "--saturation",
        type=_saturation_float,
        default=1.0,
        help="Color saturation multiplier. Range: 0 to 3.",
    )
    trail_group = parser.add_mutually_exclusive_group()
    trail_group.add_argument(
        "--trail-accumulation-frames",
        "--trail-accumulation",
        dest="trail_accumulation",
        type=_non_negative_int,
        default=0,
        help=(
            "Temporal trail accumulation amount in frames (0 disables). "
            "--trail-accumulation is an alias."
        ),
    )
    trail_group.add_argument(
        "--trail-accumulation-seconds",
        type=_non_negative_float,
        default=None,
        help="Temporal trail accumulation amount in seconds.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec, e.g. mp4v, avc1, XVID.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS (default: input FPS, or 30 if unavailable).",
    )
    args = parser.parse_args(argv)
    if args.freeze_frame is not None:
        if args.rgb_offset_frames is not None or args.rgb_offset_seconds is not None:
            parser.error(
                "--freeze-frame cannot be combined with --rgb-offset-frames or --rgb-offset-seconds"
            )
        if args.overlay_original:
            parser.error("--freeze-frame cannot be combined with --overlay-original")
    return args


def _load_frozen_reference(input_path: Path, frame_index: int, pre_blur: int) -> np.ndarray:
    if frame_index < 0:
        raise ValueError("--freeze-frame must be >= 0")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    try:
        # Fast path: jump directly to target frame when backend supports seeking.
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"--freeze-frame {frame_index} is beyond video length")

        # Some backends may ignore seek requests; fall back to sequential scan.
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_index > 0 and pos not in (frame_index, frame_index + 1):
            cap.release()
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open input video: {input_path}")
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(
                        f"--freeze-frame {frame_index} is beyond video length"
                    )
                if i == frame_index:
                    break
                i += 1

        if pre_blur > 0:
            frame = cv2.GaussianBlur(frame, (pre_blur, pre_blur), 0)
        return frame
    finally:
        cap.release()


def _reference_for_offset(history: deque[np.ndarray], offset: int) -> np.ndarray:
    # history[-1] is current frame; history[-1 - offset] is the delayed frame.
    index = len(history) - 1 - offset
    if index < 0:
        index = 0
    return history[index]


def _iter_ffmpeg_candidate_paths() -> list[Path]:
    names = ["ffmpeg"]
    candidates: list[Path] = []

    env_value = os.environ.get(FFMPEG_PATH_ENV, "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        meipass_dir = Path(str(meipass))
        for name in names:
            candidates.append(meipass_dir / name)
            candidates.append(meipass_dir / "bin" / name)

    exe_dir = Path(sys.executable).resolve().parent
    for name in names:
        candidates.append(exe_dir / name)
        candidates.append(exe_dir / "bin" / name)
        if exe_dir.name == "MacOS":
            # PyInstaller macOS app bundles typically place extra binaries in Resources.
            candidates.append(exe_dir.parent / "Resources" / name)
            candidates.append(exe_dir.parent / "Resources" / "bin" / name)

    module_dir = Path(__file__).resolve().parent
    for name in names:
        candidates.append(module_dir / name)
        candidates.append(module_dir / "bin" / name)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_ffmpeg_executable() -> str | None:
    for candidate in _iter_ffmpeg_candidate_paths():
        if not candidate.is_file():
            continue
        if not os.access(candidate, os.X_OK):
            try:
                candidate.chmod(candidate.stat().st_mode | 0o111)
            except OSError:
                pass
        if os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("ffmpeg")


def _mux_audio_into_output(
    input_path: Path,
    video_only_path: Path,
    output_path: Path,
) -> None:
    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to copy audio into the output video. "
            f"Install ffmpeg, set {FFMPEG_PATH_ENV}, or use a packaged app build "
            "that bundles ffmpeg."
        )

    base_cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-shortest",
    ]
    copy_audio_cmd = [*base_cmd, "-c:a", "copy", str(output_path)]
    result = subprocess.run(copy_audio_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return

    # Fallback for sources whose original audio codec is not MP4-compatible.
    reencode_audio_cmd = [*base_cmd, "-c:a", "aac", "-b:a", "192k", str(output_path)]
    result_fallback = subprocess.run(reencode_audio_cmd, capture_output=True, text=True)
    if result_fallback.returncode == 0:
        return

    raise RuntimeError(
        "Failed to mux audio into output.\n"
        f"ffmpeg copy-audio error:\n{result.stderr.strip()}\n\n"
        f"ffmpeg AAC fallback error:\n{result_fallback.stderr.strip()}"
    )


def process_video(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    writer = None
    async_writer = None
    video_only_output: Path | None = None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read video dimensions")

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = args.fps if args.fps is not None else (input_fps if input_fps > 0 else 30.0)

        if args.offset_seconds is not None:
            offset_frames = int(round(args.offset_seconds * fps))
        elif args.offset_frames is not None:
            offset_frames = args.offset_frames
        else:
            offset_frames = 1
        if offset_frames < 0:
            raise ValueError("offset must be >= 0")

        effective_rgb_offsets = resolve_rgb_offsets_frames(
            args.rgb_offset_frames, args.rgb_offset_seconds, fps
        )

        rgb_offsets_bgr = None
        if effective_rgb_offsets is not None:
            if args.freeze_frame is not None:
                raise ValueError(
                    "--freeze-frame cannot be combined with --rgb-offset-frames/--rgb-offset-seconds"
                )
            red, green, blue = effective_rgb_offsets
            # OpenCV channel order is BGR.
            rgb_offsets_bgr = (blue, green, red)
            max_offset = max(rgb_offsets_bgr)
        else:
            max_offset = offset_frames

        if args.output is None:
            args.output = build_default_output_path(
                args.input,
                effective_rgb_offsets,
                offset_frames,
                args.freeze_frame,
                args.overlay_original,
            )

        contrast_lut = _build_contrast_lut(PROCESS_MODE, args.contrast)
        overlay_highlight_lut = (
            _build_overlay_highlight_lut(PROCESS_MODE)
            if args.overlay_original
            else None
        )
        blur_kernel = (args.pre_blur, args.pre_blur) if args.pre_blur > 0 else None
        apply_saturation = (not args.grayscale) and args.saturation != 1.0
        saturation_lut = (
            np.clip(np.arange(256, dtype=np.float32) * args.saturation, 0, 255)
            .astype(np.uint8)
            if apply_saturation
            else None
        )
        trail_decay = None
        trail_accumulation_frames = resolve_trail_accumulation_frames(
            args.trail_accumulation, args.trail_accumulation_seconds, fps
        )
        if trail_accumulation_frames > 0:
            trail_decay = trail_accumulation_frames / (trail_accumulation_frames + 1.0)

        frame_shape = (height, width, 3)
        invert_buffer = (
            np.empty(frame_shape, dtype=np.uint8) if PROCESS_MODE == "invert50" else None
        )
        sum_u16_buffer = (
            np.empty(frame_shape, dtype=np.uint16) if PROCESS_MODE == "invert50" else None
        )
        gray_buffer = np.empty((height, width), dtype=np.uint8) if args.grayscale else None
        mixed_reference_buffer = (
            np.empty(frame_shape, dtype=np.uint8) if rgb_offsets_bgr is not None else None
        )
        saturation_hsv_buffer = (
            np.empty(frame_shape, dtype=np.uint8) if apply_saturation else None
        )
        trail_accumulation_buffer = (
            np.zeros(frame_shape, dtype=np.float32) if trail_decay is not None else None
        )
        trail_output_float_buffer = (
            np.empty(frame_shape, dtype=np.float32) if trail_decay is not None else None
        )
        overlay_highlight_buffer = (
            np.empty(frame_shape, dtype=np.uint8) if args.overlay_original else None
        )
        overlay_orig_u16_buffer = (
            np.empty(frame_shape, dtype=np.uint16) if args.overlay_original else None
        )
        overlay_high_u16_buffer = (
            np.empty(frame_shape, dtype=np.uint16) if args.overlay_original else None
        )
        overlay_inv_prod_u16_buffer = (
            np.empty(frame_shape, dtype=np.uint16) if args.overlay_original else None
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"{args.output.stem}.",
            suffix=".video_only.mp4",
            dir=str(args.output.parent),
            delete=False,
        ) as tmp_file:
            video_only_output = Path(tmp_file.name)
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        writer = cv2.VideoWriter(str(video_only_output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open output video writer: {args.output} (codec={args.codec})"
            )
        async_writer = _AsyncVideoWritePool(writer, frame_shape=frame_shape, pool_size=4)

        frozen_reference = None
        if args.freeze_frame is not None:
            frozen_reference = _load_frozen_reference(
                args.input, args.freeze_frame, args.pre_blur
            )

        history = deque(maxlen=max(max_offset + 1, 1))
        written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            base_frame = frame
            motion_frame = frame
            if blur_kernel is not None:
                motion_frame = cv2.GaussianBlur(motion_frame, blur_kernel, 0)

            if frozen_reference is None:
                history.append(motion_frame)

            write_index, output_buffer = async_writer.acquire()

            if frozen_reference is not None:
                out = compute_motion_frame(
                    current=motion_frame,
                    reference=frozen_reference,
                    mode=PROCESS_MODE,
                    contrast=args.contrast,
                    grayscale=args.grayscale,
                    contrast_lut=contrast_lut,
                    dst=output_buffer,
                    invert_buffer=invert_buffer,
                    sum_u16_buffer=sum_u16_buffer,
                    gray_buffer=gray_buffer,
                )
            else:
                if rgb_offsets_bgr is not None:
                    out = compute_motion_frame_rgb_offsets(
                        current=motion_frame,
                        reference_frames_bgr=(
                            _reference_for_offset(history, rgb_offsets_bgr[0]),
                            _reference_for_offset(history, rgb_offsets_bgr[1]),
                            _reference_for_offset(history, rgb_offsets_bgr[2]),
                        ),
                        mode=PROCESS_MODE,
                        contrast=args.contrast,
                        grayscale=args.grayscale,
                        contrast_lut=contrast_lut,
                        mixed_reference_buffer=mixed_reference_buffer,
                        dst=output_buffer,
                        invert_buffer=invert_buffer,
                        sum_u16_buffer=sum_u16_buffer,
                        gray_buffer=gray_buffer,
                    )
                else:
                    out = compute_motion_frame(
                        current=motion_frame,
                        reference=_reference_for_offset(history, offset_frames),
                        mode=PROCESS_MODE,
                        contrast=args.contrast,
                        grayscale=args.grayscale,
                        contrast_lut=contrast_lut,
                        dst=output_buffer,
                        invert_buffer=invert_buffer,
                        sum_u16_buffer=sum_u16_buffer,
                        gray_buffer=gray_buffer,
                    )
            if apply_saturation:
                _apply_saturation_bgr_in_place(
                    out,
                    args.saturation,
                    saturation_hsv_buffer,
                    saturation_lut,
                )
            if trail_decay is not None:
                _accumulate_trails_in_place(
                    out,
                    trail_accumulation_buffer,
                    trail_output_float_buffer,
                    trail_decay,
                    PROCESS_MODE,
                )
            if args.overlay_original:
                out = overlay_motion_on_original(
                    base_frame,
                    out,
                    PROCESS_MODE,
                    overlay_strength=args.overlay_strength,
                    highlight_lut=overlay_highlight_lut,
                    highlight_buffer=overlay_highlight_buffer,
                    orig_u16_buffer=overlay_orig_u16_buffer,
                    high_u16_buffer=overlay_high_u16_buffer,
                    inv_prod_u16_buffer=overlay_inv_prod_u16_buffer,
                    out_u8_buffer=output_buffer,
                )
            async_writer.submit(write_index)
            written += 1

        async_writer.close()
        async_writer = None
        writer.release()
        writer = None
        if video_only_output is None:
            raise RuntimeError("Internal error: temporary output path was not created")
        _mux_audio_into_output(args.input, video_only_output, args.output)
        if video_only_output.exists():
            video_only_output.unlink()
        return written
    finally:
        if async_writer is not None:
            # Safe to call close multiple times only if worker already joined;
            # guard by checking thread liveness.
            if async_writer.worker.is_alive():
                async_writer.close()
        if writer is not None:
            writer.release()
        cap.release()
        if video_only_output is not None and video_only_output.exists():
            video_only_output.unlink()


def main() -> None:
    args = parse_args()
    frames = process_video(args)
    print(f"Wrote {frames} frames to {args.output}")


if __name__ == "__main__":
    main()
