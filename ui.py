#!/usr/bin/env python3
"""Desktop UI for motion_extraction.py."""

from __future__ import annotations

import contextlib
import io
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from motion_extraction import default_output_directory, parse_args, process_video


class MotionExtractionUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Motion Extraction UI")
        self.root.geometry("980x760")

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.is_running = False

        self.input_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="")
        self.input_display_var = tk.StringVar(value="(No input selected)")
        self.output_display_var = tk.StringVar(value="(System default)")
        self.offset_kind_var = tk.StringVar(value="frames")
        self.offset_frames_var = tk.StringVar(value="1")
        self.offset_seconds_var = tk.StringVar(value="1.0")

        self.rgb_enabled_var = tk.BooleanVar(value=False)
        self.rgb_kind_var = tk.StringVar(value="frames")
        self.rgb_frames_r_var = tk.StringVar(value="5")
        self.rgb_frames_g_var = tk.StringVar(value="10")
        self.rgb_frames_b_var = tk.StringVar(value="15")
        self.rgb_seconds_r_var = tk.StringVar(value="0.1")
        self.rgb_seconds_g_var = tk.StringVar(value="0.2")
        self.rgb_seconds_b_var = tk.StringVar(value="0.3")

        self.freeze_enabled_var = tk.BooleanVar(value=False)
        self.freeze_frame_var = tk.StringVar(value="0")
        self.contrast_var = tk.StringVar(value="1.0")
        self.pre_blur_var = tk.StringVar(value="0")
        self.overlay_strength_var = tk.StringVar(value="1.0")
        self.saturation_var = tk.StringVar(value="1.0")
        self.trail_enabled_var = tk.BooleanVar(value=False)
        self.trail_kind_var = tk.StringVar(value="frames")
        self.trail_frames_var = tk.StringVar(value="0")
        self.trail_seconds_var = tk.StringVar(value="0.0")
        self.grayscale_var = tk.BooleanVar(value=True)
        self.overlay_var = tk.BooleanVar(value=False)
        self.preset_var = tk.StringVar(value="Pure Motion Extraction")
        self.input_fps_var = tk.StringVar(value="Input FPS: -")
        self._last_fps_probe_path: str | None = None

        self.presets: dict[str, dict[str, object]] = {
            "Pure Motion Extraction": {
                "rgb_enabled": False,
                "freeze_enabled": False,
                "offset_kind": "frames",
                "offset_frames": "1",
                "offset_seconds": "1.0",
                "rgb_kind": "frames",
                "rgb_frames": "5,10,15",
                "rgb_seconds": "0.1,0.2,0.3",
                "freeze_frame": "",
                "contrast": "1.0",
                "pre_blur": "0",
                "overlay_strength": "1.0",
                "saturation": "1.0",
                "trail_enabled": False,
                "trail_kind": "frames",
                "trail_frames": "0",
                "trail_seconds": "0.0",
                "overlay": False,
                "grayscale": True,
            },
            "Blurry RGB": {
                "rgb_enabled": True,
                "freeze_enabled": False,
                "offset_kind": "frames",
                "offset_frames": "1",
                "rgb_kind": "seconds",
                "rgb_frames": "5,10,15",
                "rgb_seconds": "0.5,1,1.5",
                "overlay": True,
                "grayscale": False,
                "overlay_strength": "1.0",
                "contrast": "1.5",
                "saturation": "2.0",
                "pre_blur": "49",
                "trail_enabled": False,
                "trail_kind": "frames",
                "trail_frames": "0",
                "trail_seconds": "0.0",
            },
            "RGB Trail": {
                "rgb_enabled": True,
                "freeze_enabled": False,
                "offset_kind": "frames",
                "offset_frames": "1",
                "rgb_kind": "seconds",
                "rgb_frames": "5,10,15",
                "rgb_seconds": "0.5,1,1.5",
                "overlay": True,
                "grayscale": False,
                "overlay_strength": "0.5",
                "contrast": "1.0",
                "saturation": "1.0",
                "pre_blur": "0",
                "trail_enabled": True,
                "trail_kind": "seconds",
                "trail_frames": "50",
                "trail_seconds": "3",
            },
        }

        self.command_preview_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self._preview_update_scheduled = False

        self._build_ui()
        self._bind_updates()
        self._refresh_path_displays()
        self._update_input_video_fps()
        self._update_dynamic_state()
        self._update_command_preview()
        self._poll_events()

    @staticmethod
    def _parse_triplet_value(value: object) -> tuple[str, str, str]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
        elif isinstance(value, (list, tuple)):
            parts = [str(part).strip() for part in value]
        else:
            return ("", "", "")
        if len(parts) != 3:
            return ("", "", "")
        return parts[0], parts[1], parts[2]

    @staticmethod
    def _collect_triplet_values(
        r_value: str,
        g_value: str,
        b_value: str,
        field_name: str,
        allow_all_empty: bool = False,
    ) -> str | None:
        r_text = r_value.strip()
        g_text = g_value.strip()
        b_text = b_value.strip()
        if not r_text and not g_text and not b_text:
            if allow_all_empty:
                return None
            raise ValueError(f"{field_name} requires R, G, and B values.")
        if not (r_text and g_text and b_text):
            raise ValueError(f"{field_name} requires R, G, and B values.")
        return f"{r_text},{g_text},{b_text}"

    @staticmethod
    def _validate_pre_blur_input(proposed: str) -> bool:
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        return int(proposed) <= 49

    @staticmethod
    def _validate_contrast_input(proposed: str) -> bool:
        if proposed == "":
            return True
        if proposed.startswith("-"):
            return False
        if proposed.count(".") > 1:
            return False
        try:
            value = float(proposed)
        except ValueError:
            return False
        return 0.0 <= value <= 10.0

    @staticmethod
    def _validate_overlay_strength_input(proposed: str) -> bool:
        if proposed == "":
            return True
        if proposed.startswith("-"):
            return False
        if proposed.count(".") > 1:
            return False
        try:
            value = float(proposed)
        except ValueError:
            return False
        return 0.0 <= value <= 1.0

    @staticmethod
    def _validate_saturation_input(proposed: str) -> bool:
        if proposed == "":
            return True
        if proposed.startswith("-"):
            return False
        if proposed.count(".") > 1:
            return False
        try:
            value = float(proposed)
        except ValueError:
            return False
        return 0.0 <= value <= 3.0

    @staticmethod
    def _validate_trail_frames_input(proposed: str) -> bool:
        if proposed == "":
            return True
        if not proposed.isdigit():
            return False
        return True

    @staticmethod
    def _validate_non_negative_float_input(proposed: str) -> bool:
        if proposed == "":
            return True
        if proposed.startswith("-"):
            return False
        if proposed.count(".") > 1:
            return False
        try:
            value = float(proposed)
        except ValueError:
            return False
        return value >= 0.0

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        validate_pre_blur_cmd = (self.root.register(self._validate_pre_blur_input), "%P")
        validate_contrast_cmd = (self.root.register(self._validate_contrast_input), "%P")
        validate_overlay_strength_cmd = (
            self.root.register(self._validate_overlay_strength_input),
            "%P",
        )
        validate_saturation_cmd = (self.root.register(self._validate_saturation_input), "%P")
        validate_trail_frames_cmd = (
            self.root.register(self._validate_trail_frames_input),
            "%P",
        )
        validate_non_negative_float_cmd = (
            self.root.register(self._validate_non_negative_float_input),
            "%P",
        )

        io_frame = ttk.LabelFrame(container, text="Input / Output", padding=10)
        io_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(io_frame, text="Input video").grid(row=0, column=0, sticky="w")
        ttk.Label(io_frame, textvariable=self.input_fps_var).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Button(io_frame, text="Browse...", command=self._choose_input).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(
            io_frame,
            textvariable=self.input_display_var,
            anchor="w",
        ).grid(
            row=1, column=1, sticky="w", padx=(8, 0)
        )

        ttk.Label(io_frame, text="Output video (optional)").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(io_frame, text="Save As...", command=self._choose_output).grid(
            row=3, column=0, sticky="w"
        )
        ttk.Label(
            io_frame,
            textvariable=self.output_display_var,
            anchor="w",
        ).grid(
            row=3, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Label(io_frame, text="Preset").grid(row=4, column=0, sticky="w", pady=(8, 0))
        preset_row = ttk.Frame(io_frame)
        preset_row.grid(row=5, column=0, columnspan=2, sticky="w")
        self.preset_combo = ttk.Combobox(
            preset_row,
            textvariable=self.preset_var,
            values=list(self.presets.keys()),
            state="readonly",
            width=20,
        )
        self.preset_combo.pack(side=tk.LEFT)
        ttk.Button(
            preset_row,
            text="Apply",
            command=self._apply_selected_preset,
        ).pack(side=tk.LEFT, padx=(8, 0))
        io_frame.columnconfigure(1, weight=1)

        mode_frame = ttk.LabelFrame(container, text="Core Options", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.rgb_toggle = ttk.Checkbutton(
            mode_frame, text="RGB Offset", variable=self.rgb_enabled_var
        )
        self.rgb_toggle.grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.overlay_toggle = ttk.Checkbutton(
            mode_frame, text="Overlay Original", variable=self.overlay_var
        )
        self.overlay_toggle.grid(row=0, column=1, sticky="w", padx=(0, 20))
        ttk.Checkbutton(
            mode_frame, text="Grayscale Output", variable=self.grayscale_var
        ).grid(row=0, column=2, sticky="w", padx=(0, 20))
        self.freeze_toggle = ttk.Checkbutton(
            mode_frame, text="Freeze Frame", variable=self.freeze_enabled_var
        )
        self.freeze_toggle.grid(row=0, column=3, sticky="w")

        self.mode_specific_container = ttk.Frame(container)
        self.mode_specific_container.pack(fill=tk.X, pady=(0, 10))
        self.mode_specific_container.columnconfigure(0, weight=1)

        self.non_rgb_frame = ttk.LabelFrame(
            self.mode_specific_container, text="Non-RGB Offset", padding=10
        )

        self.non_rgb_frames_radio = ttk.Radiobutton(
            self.non_rgb_frame,
            text="Frames",
            variable=self.offset_kind_var,
            value="frames",
        )
        self.non_rgb_frames_radio.grid(row=0, column=0, sticky="w")
        self.non_rgb_frames_entry = ttk.Entry(
            self.non_rgb_frame,
            textvariable=self.offset_frames_var,
            width=12,
        )
        self.non_rgb_frames_entry.grid(row=0, column=1, sticky="w", padx=(8, 20))

        self.non_rgb_seconds_radio = ttk.Radiobutton(
            self.non_rgb_frame,
            text="Seconds",
            variable=self.offset_kind_var,
            value="seconds",
        )
        self.non_rgb_seconds_radio.grid(row=0, column=2, sticky="w")
        self.non_rgb_seconds_entry = ttk.Entry(
            self.non_rgb_frame,
            textvariable=self.offset_seconds_var,
            width=12,
        )
        self.non_rgb_seconds_entry.grid(row=0, column=3, sticky="w", padx=(8, 0))

        self.rgb_frame = ttk.LabelFrame(self.mode_specific_container, text="RGB Offset", padding=10)

        self.rgb_frames_radio = ttk.Radiobutton(
            self.rgb_frame,
            text="Frames (R,G,B)",
            variable=self.rgb_kind_var,
            value="frames",
        )
        self.rgb_frames_radio.grid(row=0, column=0, sticky="w")
        ttk.Label(self.rgb_frame, text="R").grid(row=0, column=1, sticky="w", padx=(8, 2))
        self.rgb_frames_r_entry = ttk.Entry(
            self.rgb_frame,
            textvariable=self.rgb_frames_r_var,
            width=6,
        )
        self.rgb_frames_r_entry.grid(row=0, column=2, sticky="w")
        ttk.Label(self.rgb_frame, text="G").grid(row=0, column=3, sticky="w", padx=(8, 2))
        self.rgb_frames_g_entry = ttk.Entry(
            self.rgb_frame,
            textvariable=self.rgb_frames_g_var,
            width=6,
        )
        self.rgb_frames_g_entry.grid(row=0, column=4, sticky="w")
        ttk.Label(self.rgb_frame, text="B").grid(row=0, column=5, sticky="w", padx=(8, 2))
        self.rgb_frames_b_entry = ttk.Entry(
            self.rgb_frame,
            textvariable=self.rgb_frames_b_var,
            width=6,
        )
        self.rgb_frames_b_entry.grid(row=0, column=6, sticky="w", padx=(0, 20))

        self.rgb_seconds_radio = ttk.Radiobutton(
            self.rgb_frame,
            text="Seconds (R,G,B)",
            variable=self.rgb_kind_var,
            value="seconds",
        )
        self.rgb_seconds_radio.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(self.rgb_frame, text="R").grid(
            row=1, column=1, sticky="w", padx=(8, 2), pady=(6, 0)
        )
        self.rgb_seconds_r_entry = ttk.Entry(
            self.rgb_frame,
            textvariable=self.rgb_seconds_r_var,
            width=6,
        )
        self.rgb_seconds_r_entry.grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Label(self.rgb_frame, text="G").grid(
            row=1, column=3, sticky="w", padx=(8, 2), pady=(6, 0)
        )
        self.rgb_seconds_g_entry = ttk.Entry(
            self.rgb_frame,
            textvariable=self.rgb_seconds_g_var,
            width=6,
        )
        self.rgb_seconds_g_entry.grid(row=1, column=4, sticky="w", pady=(6, 0))
        ttk.Label(self.rgb_frame, text="B").grid(
            row=1, column=5, sticky="w", padx=(8, 2), pady=(6, 0)
        )
        self.rgb_seconds_b_entry = ttk.Entry(
            self.rgb_frame,
            textvariable=self.rgb_seconds_b_var,
            width=6,
        )
        self.rgb_seconds_b_entry.grid(
            row=1, column=6, sticky="w", padx=(0, 20), pady=(6, 0)
        )

        self.freeze_frame_mode_frame = ttk.LabelFrame(
            self.mode_specific_container, text="Freeze Frame", padding=10
        )
        ttk.Label(self.freeze_frame_mode_frame, text="Frame Index").grid(
            row=0, column=0, sticky="w"
        )
        self.freeze_entry = ttk.Entry(
            self.freeze_frame_mode_frame, textvariable=self.freeze_frame_var, width=12
        )
        self.freeze_entry.grid(row=0, column=1, sticky="w", padx=(8, 20))
        ttk.Label(
            self.freeze_frame_mode_frame,
            text="Uses one fixed frame as the reference for the entire output.",
        ).grid(row=0, column=2, sticky="w")

        self.non_rgb_frame.grid(row=0, column=0, sticky="ew")
        self.rgb_frame.grid(row=0, column=0, sticky="ew")
        self.freeze_frame_mode_frame.grid(row=0, column=0, sticky="ew")

        self.adv_frame = ttk.LabelFrame(container, text="Advanced", padding=10)
        self.adv_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.adv_frame, text="Contrast (0-10)").grid(
            row=0, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            self.adv_frame,
            textvariable=self.contrast_var,
            width=12,
            validate="key",
            validatecommand=validate_contrast_cmd,
        ).grid(row=0, column=1, sticky="w", padx=(8, 20), pady=(8, 0))

        ttk.Label(self.adv_frame, text="Saturation (0-3)").grid(
            row=0, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            self.adv_frame,
            textvariable=self.saturation_var,
            width=12,
            validate="key",
            validatecommand=validate_saturation_cmd,
        ).grid(row=0, column=3, sticky="w", padx=(8, 20), pady=(8, 0))

        ttk.Label(self.adv_frame, text="Overlay Strength (0.0-1.0)").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            self.adv_frame,
            textvariable=self.overlay_strength_var,
            width=12,
            validate="key",
            validatecommand=validate_overlay_strength_cmd,
        ).grid(row=1, column=1, sticky="w", padx=(8, 20), pady=(8, 0))

        ttk.Label(self.adv_frame, text="Pre-Blur (odd 0-49)").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            self.adv_frame,
            textvariable=self.pre_blur_var,
            width=12,
            validate="key",
            validatecommand=validate_pre_blur_cmd,
        ).grid(row=1, column=3, sticky="w", padx=(8, 20), pady=(8, 0))

        self.trail_toggle = ttk.Checkbutton(
            self.adv_frame,
            text="Trail Accumulation",
            variable=self.trail_enabled_var,
        )
        self.trail_toggle.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.trail_frame = ttk.Frame(self.adv_frame)
        self.trail_frame.grid(row=2, column=1, columnspan=3, sticky="w", pady=(8, 0))
        self.trail_frames_radio = ttk.Radiobutton(
            self.trail_frame,
            text="Frames",
            variable=self.trail_kind_var,
            value="frames",
        )
        self.trail_frames_radio.grid(row=0, column=0, sticky="w")
        self.trail_frames_entry = ttk.Entry(
            self.trail_frame,
            textvariable=self.trail_frames_var,
            width=12,
            validate="key",
            validatecommand=validate_trail_frames_cmd,
        )
        self.trail_frames_entry.grid(row=0, column=1, sticky="w", padx=(8, 16))
        self.trail_seconds_radio = ttk.Radiobutton(
            self.trail_frame,
            text="Seconds",
            variable=self.trail_kind_var,
            value="seconds",
        )
        self.trail_seconds_radio.grid(row=0, column=2, sticky="w")
        self.trail_seconds_entry = ttk.Entry(
            self.trail_frame,
            textvariable=self.trail_seconds_var,
            width=12,
            validate="key",
            validatecommand=validate_non_negative_float_cmd,
        )
        self.trail_seconds_entry.grid(row=0, column=3, sticky="w", padx=(8, 0))

        cmd_frame = ttk.LabelFrame(container, text="Command Preview", padding=10)
        cmd_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(
            cmd_frame,
            textvariable=self.command_preview_var,
            state="readonly",
        ).pack(fill=tk.X)

        action_frame = ttk.Frame(container)
        action_frame.pack(fill=tk.X, pady=(0, 8))

        self.run_button = ttk.Button(action_frame, text="Run", command=self._run_clicked)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(action_frame, text="Copy Command", command=self._copy_command).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        self.progress = ttk.Progressbar(action_frame, mode="indeterminate", length=220)
        self.progress.pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(action_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(10, 0))

    def _apply_selected_preset(self) -> None:
        preset_name = self.preset_var.get().strip()
        preset = self.presets.get(preset_name)
        if preset is None:
            messagebox.showerror("Invalid preset", f"Preset not found: {preset_name}")
            return

        if "rgb_enabled" in preset:
            self.rgb_enabled_var.set(bool(preset["rgb_enabled"]))
        if "freeze_enabled" in preset:
            self.freeze_enabled_var.set(bool(preset["freeze_enabled"]))
        if "offset_kind" in preset:
            self.offset_kind_var.set(str(preset["offset_kind"]))
        if "offset_frames" in preset:
            self.offset_frames_var.set(str(preset["offset_frames"]))
        if "offset_seconds" in preset:
            self.offset_seconds_var.set(str(preset["offset_seconds"]))
        if "rgb_kind" in preset:
            self.rgb_kind_var.set(str(preset["rgb_kind"]))
        if "rgb_frames" in preset:
            r, g, b = self._parse_triplet_value(preset["rgb_frames"])
            self.rgb_frames_r_var.set(r)
            self.rgb_frames_g_var.set(g)
            self.rgb_frames_b_var.set(b)
        if "rgb_seconds" in preset:
            r, g, b = self._parse_triplet_value(preset["rgb_seconds"])
            self.rgb_seconds_r_var.set(r)
            self.rgb_seconds_g_var.set(g)
            self.rgb_seconds_b_var.set(b)
        if "freeze_frame" in preset:
            self.freeze_frame_var.set(str(preset["freeze_frame"]))
        if "contrast" in preset:
            self.contrast_var.set(str(preset["contrast"]))
        if "pre_blur" in preset:
            self.pre_blur_var.set(str(preset["pre_blur"]))
        if "overlay_strength" in preset:
            self.overlay_strength_var.set(str(preset["overlay_strength"]))
        if "saturation" in preset:
            self.saturation_var.set(str(preset["saturation"]))
        if "trail_enabled" in preset:
            self.trail_enabled_var.set(bool(preset["trail_enabled"]))
        if "trail_kind" in preset:
            self.trail_kind_var.set(str(preset["trail_kind"]))
        if "trail_frames" in preset:
            self.trail_frames_var.set(str(preset["trail_frames"]))
        if "trail_seconds" in preset:
            self.trail_seconds_var.set(str(preset["trail_seconds"]))
        if "overlay" in preset:
            self.overlay_var.set(bool(preset["overlay"]))
        if "grayscale" in preset:
            self.grayscale_var.set(bool(preset["grayscale"]))

    def _bind_updates(self) -> None:
        self.input_var.trace_add("write", self._on_input_changed)
        tracked = [
            self.output_var,
            self.offset_kind_var,
            self.offset_frames_var,
            self.offset_seconds_var,
            self.rgb_enabled_var,
            self.freeze_enabled_var,
            self.rgb_kind_var,
            self.rgb_frames_r_var,
            self.rgb_frames_g_var,
            self.rgb_frames_b_var,
            self.rgb_seconds_r_var,
            self.rgb_seconds_g_var,
            self.rgb_seconds_b_var,
            self.freeze_frame_var,
            self.contrast_var,
            self.pre_blur_var,
            self.overlay_strength_var,
            self.saturation_var,
            self.trail_enabled_var,
            self.trail_kind_var,
            self.trail_frames_var,
            self.trail_seconds_var,
            self.grayscale_var,
            self.overlay_var,
        ]
        for var in tracked:
            var.trace_add("write", self._on_inputs_changed)

    def _on_input_changed(self, *_: object) -> None:
        self._refresh_path_displays()
        self._update_input_video_fps()
        self._schedule_command_preview_update()

    def _on_inputs_changed(self, *_: object) -> None:
        self._refresh_path_displays()
        self._update_dynamic_state()
        self._schedule_command_preview_update()

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose input video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _choose_output(self) -> None:
        start_dir = default_output_directory()
        path = filedialog.asksaveasfilename(
            title="Choose output video",
            initialdir=str(start_dir),
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _refresh_path_displays(self) -> None:
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip()
        self.input_display_var.set(input_path if input_path else "(No input selected)")
        self.output_display_var.set(
            output_path if output_path else "(System default)"
        )

    def _update_dynamic_state(self) -> None:
        rgb_enabled = self.rgb_enabled_var.get()
        freeze_enabled = self.freeze_enabled_var.get()

        if freeze_enabled:
            if rgb_enabled:
                self.rgb_enabled_var.set(False)
                rgb_enabled = False
            if self.overlay_var.get():
                self.overlay_var.set(False)
            self.rgb_toggle.configure(state="disabled")
            self.overlay_toggle.configure(state="disabled")
            self.freeze_toggle.configure(state="normal")
        else:
            self.rgb_toggle.configure(state="normal")
            self.overlay_toggle.configure(state="normal")
            self.freeze_toggle.configure(state="disabled" if rgb_enabled else "normal")

        non_rgb_frames = self.offset_kind_var.get() == "frames"
        rgb_frames = self.rgb_kind_var.get() == "frames"

        if rgb_enabled:
            self.non_rgb_frame.grid_remove()
            self.freeze_frame_mode_frame.grid_remove()
            self.rgb_frame.grid()
        elif freeze_enabled:
            self.rgb_frame.grid_remove()
            self.non_rgb_frame.grid_remove()
            self.freeze_frame_mode_frame.grid()
        else:
            self.rgb_frame.grid_remove()
            self.freeze_frame_mode_frame.grid_remove()
            self.non_rgb_frame.grid()

        self.non_rgb_frames_radio.configure(state="normal")
        self.non_rgb_seconds_radio.configure(state="normal")
        self.non_rgb_frames_entry.configure(
            state=(
                "normal"
                if non_rgb_frames
                else "disabled"
            )
        )
        self.non_rgb_seconds_entry.configure(
            state=(
                "normal"
                if not non_rgb_frames
                else "disabled"
            )
        )

        self.rgb_frames_radio.configure(state="normal")
        self.rgb_seconds_radio.configure(state="normal")
        rgb_frames_state = "normal" if rgb_frames else "disabled"
        rgb_seconds_state = "normal" if not rgb_frames else "disabled"
        self.rgb_frames_r_entry.configure(state=rgb_frames_state)
        self.rgb_frames_g_entry.configure(state=rgb_frames_state)
        self.rgb_frames_b_entry.configure(state=rgb_frames_state)
        self.rgb_seconds_r_entry.configure(state=rgb_seconds_state)
        self.rgb_seconds_g_entry.configure(state=rgb_seconds_state)
        self.rgb_seconds_b_entry.configure(state=rgb_seconds_state)

        trail_enabled = self.trail_enabled_var.get()
        trail_frames = self.trail_kind_var.get() == "frames"
        if trail_enabled:
            self.trail_frame.grid()
            self.trail_frames_radio.configure(state="normal")
            self.trail_seconds_radio.configure(state="normal")
            self.trail_frames_entry.configure(state="normal" if trail_frames else "disabled")
            self.trail_seconds_entry.configure(
                state="normal" if not trail_frames else "disabled"
            )
        else:
            self.trail_frame.grid_remove()
            self.trail_frames_radio.configure(state="disabled")
            self.trail_seconds_radio.configure(state="disabled")
            self.trail_frames_entry.configure(state="disabled")
            self.trail_seconds_entry.configure(state="disabled")

    def _build_argv(self) -> list[str]:
        input_path = self.input_var.get().strip()
        if not input_path:
            raise ValueError("Input video is required.")

        argv = [input_path]

        output_path = self.output_var.get().strip()
        if output_path:
            argv.append(output_path)

        if self.rgb_enabled_var.get():
            if self.rgb_kind_var.get() == "frames":
                rgb_frames = self._collect_triplet_values(
                    self.rgb_frames_r_var.get(),
                    self.rgb_frames_g_var.get(),
                    self.rgb_frames_b_var.get(),
                    field_name="RGB frame offsets",
                    allow_all_empty=True,
                )
                if rgb_frames is None:
                    # Uses default R5,G10,B15.
                    argv.append("--rgb-offset-frames")
                else:
                    argv.extend(["--rgb-offset-frames", rgb_frames])
            else:
                rgb_seconds = self._collect_triplet_values(
                    self.rgb_seconds_r_var.get(),
                    self.rgb_seconds_g_var.get(),
                    self.rgb_seconds_b_var.get(),
                    field_name="RGB seconds offsets",
                )
                argv.extend(["--rgb-offset-seconds", rgb_seconds])
        else:
            if self.freeze_enabled_var.get():
                freeze_frame = self.freeze_frame_var.get().strip()
                if not freeze_frame:
                    raise ValueError(
                        "Freeze Frame is enabled but no frame index was provided."
                    )
                argv.extend(["--freeze-frame", freeze_frame])
            else:
                if self.offset_kind_var.get() == "frames":
                    argv.extend(["--frame-offset", self.offset_frames_var.get().strip() or "1"])
                else:
                    value = self.offset_seconds_var.get().strip()
                    if not value:
                        raise ValueError("Offset seconds is selected but no value was provided.")
                    argv.extend(["--offset-seconds", value])

        argv.extend(["--contrast", self.contrast_var.get().strip() or "1.0"])
        argv.extend(["--pre-blur", self.pre_blur_var.get().strip() or "0"])
        argv.extend(
            ["--overlay-strength", self.overlay_strength_var.get().strip() or "1.0"]
        )
        argv.extend(["--saturation", self.saturation_var.get().strip() or "1.0"])
        if self.trail_enabled_var.get():
            if self.trail_kind_var.get() == "frames":
                argv.extend(
                    [
                        "--trail-accumulation-frames",
                        self.trail_frames_var.get().strip() or "0",
                    ]
                )
            else:
                trail_seconds = self.trail_seconds_var.get().strip() or "0.0"
                argv.extend(["--trail-accumulation-seconds", trail_seconds])

        if self.grayscale_var.get():
            argv.append("--grayscale")
        if self.overlay_var.get():
            argv.append("--overlay-original")

        return argv

    def _update_input_video_fps(self) -> None:
        input_path = self.input_var.get().strip()
        if input_path == self._last_fps_probe_path:
            return
        self._last_fps_probe_path = input_path

        if not input_path:
            self.input_fps_var.set("Input FPS: -")
            return

        path = Path(input_path)
        if not path.exists():
            self.input_fps_var.set("Input FPS: file not found")
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.input_fps_var.set("Input FPS: unreadable")
            return
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self.input_fps_var.set(f"Input FPS: {fps:.3f}")
            else:
                self.input_fps_var.set("Input FPS: unavailable")
        finally:
            cap.release()

    def _safe_parse(self, argv: list[str]):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                return parse_args(argv)
            except SystemExit as exc:
                if exc.code == 0:
                    raise ValueError("Unexpected parse exit.")
                msg = err.getvalue().strip()
                raise ValueError(msg or "Invalid arguments.")

    def _update_command_preview(self) -> None:
        try:
            argv = self._build_argv()
            # Validate preview command early.
            self._safe_parse(argv)
            cmd = "python3 motion_extraction.py " + shlex.join(argv)
            self.command_preview_var.set(cmd)
        except Exception as exc:
            self.command_preview_var.set(f"Invalid configuration: {exc}")

    def _schedule_command_preview_update(self) -> None:
        if self._preview_update_scheduled:
            return
        self._preview_update_scheduled = True
        self.root.after_idle(self._run_scheduled_command_preview_update)

    def _run_scheduled_command_preview_update(self) -> None:
        self._preview_update_scheduled = False
        self._update_command_preview()

    def _copy_command(self) -> None:
        cmd = self.command_preview_var.get()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)

    def _run_clicked(self) -> None:
        if self.is_running:
            return
        try:
            argv = self._build_argv()
            parsed = self._safe_parse(argv)
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")

        self.worker = threading.Thread(
            target=self._worker_run,
            args=(parsed,),
            daemon=True,
        )
        self.worker.start()

    def _worker_run(self, parsed_args) -> None:
        start = time.time()
        try:
            frames = process_video(parsed_args)
            elapsed = time.time() - start
            self.events.put(
                (
                    "success",
                    {
                        "frames": frames,
                        "elapsed": elapsed,
                        "output": str(parsed_args.output),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - runtime behavior
            self.events.put(("error", str(exc)))
        finally:
            self.events.put(("done", None))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "success":
                data = payload
                assert isinstance(data, dict)
                self.status_var.set("Done")
                self._open_output_video(str(data["output"]))
            elif kind == "error":
                self.status_var.set("Error")
                messagebox.showerror("Processing error", str(payload))
            elif kind == "done":
                self.is_running = False
                self.run_button.configure(state="normal")
                self.progress.stop()

        self.root.after(100, self._poll_events)

    def _open_output_video(self, output_path: str) -> None:
        path = Path(output_path)
        if not path.exists():
            self.status_var.set("Done (output not found)")
            return

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # pragma: no cover - environment dependent
            self.status_var.set(f"Done (open failed: {exc})")


def main() -> None:
    root = tk.Tk()
    MotionExtractionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
