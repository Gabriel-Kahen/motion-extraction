import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from motion_extraction import (
    build_default_output_path,
    compute_motion_frame,
    compute_motion_frame_rgb_offsets,
    default_output_directory,
    overlay_motion_on_original,
    parse_args,
    resolve_ffmpeg_executable,
    resolve_rgb_offsets_frames,
    resolve_trail_accumulation_frames,
)


class MotionExtractionTests(unittest.TestCase):
    def test_parse_args_defaults_to_video_and_auto_output(self) -> None:
        args = parse_args([])
        self.assertEqual(args.input, Path("videos/video.mp4"))
        self.assertIsNone(args.output)

    def test_build_default_output_path_non_rgb(self) -> None:
        path = build_default_output_path(Path("video.mp4"), None, 12, None, False)
        self.assertEqual(path, default_output_directory() / "video_output_f12.mp4")

    def test_build_default_output_path_rgb(self) -> None:
        path = build_default_output_path(
            Path("clip.mov"), (1, 10, 20), 1, None, False
        )
        self.assertEqual(
            path, default_output_directory() / "clip_output_rgb_r1_g10_b20.mp4"
        )

    def test_build_default_output_path_appends_overlaid_when_overlay_enabled(self) -> None:
        path = build_default_output_path(Path("rain.mp4"), None, 3, None, True)
        self.assertEqual(
            path, default_output_directory() / "rain_output_f3_overlaid.mp4"
        )

    def test_parse_args_accepts_rgb_offsets(self) -> None:
        args = parse_args(["in.mp4", "out.mp4", "--rgb-offset-frames", "1,2,3"])
        self.assertEqual(args.rgb_offset_frames, (1, 2, 3))

    def test_parse_args_accepts_rgb_offset_seconds(self) -> None:
        args = parse_args(["--rgb-offset-seconds", "0,0.5,1.25"])
        self.assertEqual(args.rgb_offset_seconds, (0.0, 0.5, 1.25))

    def test_parse_args_uses_default_rgb_offsets_when_flag_has_no_value(self) -> None:
        args = parse_args(["--rgb-offset-frames"])
        self.assertEqual(args.rgb_offset_frames, (5, 10, 15))

    def test_parse_args_rejects_non_rgb_frames_and_seconds_together(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--frame-offset", "7", "--offset-seconds", "0.2"])

    def test_parse_args_rejects_rgb_frames_and_seconds_together(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                ["--rgb-offset-frames", "1,2,3", "--rgb-offset-seconds", "0,0.5,1.25"]
            )

    def test_parse_args_rejects_freeze_frame_with_rgb_offsets(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--freeze-frame", "0", "--rgb-offset-frames", "1,2,3"])

    def test_parse_args_rejects_freeze_frame_with_overlay_original(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--freeze-frame", "0", "--overlay-original"])

    def test_resolve_rgb_offsets_frames_from_seconds(self) -> None:
        offsets = resolve_rgb_offsets_frames(None, (0.0, 0.5, 1.25), 20.0)
        self.assertEqual(offsets, (0, 10, 25))

    def test_resolve_rgb_offsets_rejects_frames_and_seconds_together(self) -> None:
        with self.assertRaises(ValueError):
            resolve_rgb_offsets_frames((5, 10, 15), (0.0, 0.5, 1.25), 20.0)

    def test_resolve_trail_accumulation_frames_from_seconds(self) -> None:
        frames = resolve_trail_accumulation_frames(0, 0.5, 20.0)
        self.assertEqual(frames, 10)

    def test_parse_args_accepts_non_rgb_frame_offset_alias(self) -> None:
        args = parse_args(["--frame-offset", "7"])
        self.assertEqual(args.offset_frames, 7)
        self.assertIsNone(args.rgb_offset_frames)

    def test_parse_args_accepts_overlay_flag(self) -> None:
        args = parse_args(["--overlay-original"])
        self.assertTrue(args.overlay_original)

    def test_parse_args_accepts_new_advanced_options(self) -> None:
        args = parse_args(
            [
                "--overlay-strength",
                "0.65",
                "--contrast",
                "3.0",
                "--saturation",
                "1.8",
                "--trail-accumulation-frames",
                "12",
            ]
        )
        self.assertAlmostEqual(args.overlay_strength, 0.65)
        self.assertAlmostEqual(args.contrast, 3.0)
        self.assertAlmostEqual(args.saturation, 1.8)
        self.assertEqual(args.trail_accumulation, 12)

    def test_parse_args_accepts_trail_accumulation_seconds(self) -> None:
        args = parse_args(["--trail-accumulation-seconds", "0.75"])
        self.assertAlmostEqual(args.trail_accumulation_seconds, 0.75)

    def test_parse_args_rejects_overlay_strength_out_of_range(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--overlay-strength", "1.2"])

    def test_parse_args_rejects_overlay_strength_below_zero(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--overlay-strength", "-0.1"])

    def test_parse_args_rejects_contrast_above_max(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--contrast", "10.1"])

    def test_parse_args_rejects_contrast_below_zero(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--contrast", "-0.1"])

    def test_parse_args_rejects_saturation_above_max(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--saturation", "3.1"])

    def test_parse_args_rejects_saturation_below_zero(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--saturation", "-0.1"])

    def test_parse_args_rejects_trail_accumulation_frames_and_seconds_together(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--trail-accumulation-frames",
                    "12",
                    "--trail-accumulation-seconds",
                    "0.4",
                ]
            )

    def test_parse_args_rejects_trail_accumulation_below_zero(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--trail-accumulation", "-1"])

    def test_parse_args_rejects_trail_accumulation_seconds_below_zero(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--trail-accumulation-seconds", "-0.1"])

    def test_parse_args_rejects_removed_start_end_seconds_flags(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--start-seconds", "5", "--end-seconds", "8"])

    def test_parse_args_accepts_pre_blur_zero(self) -> None:
        args = parse_args(["--pre-blur", "0"])
        self.assertEqual(args.pre_blur, 0)

    def test_parse_args_rejects_even_pre_blur(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--pre-blur", "4"])

    def test_parse_args_rejects_pre_blur_above_max(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--pre-blur", "51"])

    def test_invert50_identical_frames_are_mid_gray(self) -> None:
        frame = np.full((2, 2, 3), 73, dtype=np.uint8)
        out = compute_motion_frame(frame, frame, mode="invert50", contrast=1.0)
        self.assertTrue(np.all(out == 128))

    def test_invert50_formula_matches_expected_values(self) -> None:
        current = np.array([[[255, 0, 128]]], dtype=np.uint8)
        reference = np.array([[[0, 255, 128]]], dtype=np.uint8)
        out = compute_motion_frame(current, reference, mode="invert50", contrast=1.0)
        expected = np.array([[[255, 0, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_difference_identical_frames_are_black(self) -> None:
        frame = np.full((3, 3, 3), 200, dtype=np.uint8)
        out = compute_motion_frame(frame, frame, mode="difference", contrast=1.0)
        self.assertTrue(np.all(out == 0))

    def test_difference_max_change_is_white(self) -> None:
        current = np.zeros((1, 1, 3), dtype=np.uint8)
        reference = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = compute_motion_frame(current, reference, mode="difference", contrast=1.0)
        self.assertTrue(np.all(out == 255))

    def test_rgb_offsets_uses_separate_reference_per_channel(self) -> None:
        current = np.array([[[10, 20, 30]]], dtype=np.uint8)
        ref_blue = np.array([[[0, 250, 250]]], dtype=np.uint8)
        ref_green = np.array([[[250, 100, 250]]], dtype=np.uint8)
        ref_red = np.array([[[250, 250, 200]]], dtype=np.uint8)
        out = compute_motion_frame_rgb_offsets(
            current,
            (ref_blue, ref_green, ref_red),
            mode="invert50",
            contrast=1.0,
        )
        expected = np.array([[[133, 88, 43]]], dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_overlay_invert50_neutral_motion_returns_original(self) -> None:
        original = np.array([[[10, 100, 200]]], dtype=np.uint8)
        neutral_motion = np.full((1, 1, 3), 128, dtype=np.uint8)
        out = overlay_motion_on_original(original, neutral_motion, "invert50")
        np.testing.assert_array_equal(out, original)

    def test_overlay_difference_white_motion_returns_white(self) -> None:
        original = np.array([[[10, 100, 200]]], dtype=np.uint8)
        full_motion = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = overlay_motion_on_original(original, full_motion, "difference")
        expected = np.full((1, 1, 3), 255, dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_overlay_strength_zero_returns_original(self) -> None:
        original = np.array([[[10, 100, 200]]], dtype=np.uint8)
        full_motion = np.full((1, 1, 3), 255, dtype=np.uint8)
        out = overlay_motion_on_original(
            original,
            full_motion,
            "invert50",
            overlay_strength=0.0,
        )
        np.testing.assert_array_equal(out, original)

    def test_resolve_ffmpeg_executable_prefers_candidate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ffmpeg_path = Path(tmp_dir) / "ffmpeg"
            ffmpeg_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            ffmpeg_path.chmod(0o755)

            with mock.patch(
                "motion_extraction._iter_ffmpeg_candidate_paths",
                return_value=[ffmpeg_path],
            ):
                with mock.patch(
                    "motion_extraction.shutil.which",
                    return_value="/usr/local/bin/ffmpeg",
                ):
                    self.assertEqual(
                        resolve_ffmpeg_executable(),
                        str(ffmpeg_path),
                    )

    def test_resolve_ffmpeg_executable_falls_back_to_path(self) -> None:
        with mock.patch(
            "motion_extraction._iter_ffmpeg_candidate_paths",
            return_value=[],
        ):
            with mock.patch(
                "motion_extraction.shutil.which",
                return_value="/usr/local/bin/ffmpeg",
            ):
                self.assertEqual(
                    resolve_ffmpeg_executable(),
                    "/usr/local/bin/ffmpeg",
                )

if __name__ == "__main__":
    unittest.main()
