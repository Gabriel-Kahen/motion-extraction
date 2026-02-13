# Motion Extraction

This is an implementation of the algorithm shown in Posy's video:
[Motion Extraction](https://www.youtube.com/watch?v=NSS6yAMZF78).

## Examples (Before → After)

| Input | Output |
| --- | --- |
| <img src="examples/in1.gif" width="360" /> | <img src="examples/out1.gif" width="360" /> |
| <img src="examples/in2.gif" width="360" /> | <img src="examples/out2.gif" width="360" /> |
| <img src="examples/in3.gif" width="360" /> | <img src="examples/out3.gif" width="360" /> |

Core idea:
1. Duplicate the video.
2. Invert the duplicate.
3. Blend it at 50% opacity.
4. Time-shift the duplicate.

Per-frame formula (always `invert50`):

`output = 0.5 * current + 0.5 * (1 - shifted_reference)`

## Requirements

- Python 3.10+
- `numpy`
- `opencv-python`
- `ffmpeg` (required so output always keeps/copies input audio)

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Desktop UI

Run the local desktop UI (Tkinter):

```bash
python3 ui.py
```

The UI supports core processing features:

- Input/output file selection
- Non-RGB offsets (`--frame-offset` or `--offset-seconds`)
- RGB offsets (`--rgb-offset-frames` or `--rgb-offset-seconds`)
- Overlay, grayscale, freeze-frame, contrast, pre-blur
- Overlay strength, saturation, trail accumulation
- Built-in presets (`Pure Motion Extraction`, `Blurry RGB`, `RGB Trail`)
- Live command preview and background processing

## macOS App (Bundled ffmpeg)

Build a macOS `.app` that bundles `ffmpeg`:

```bash
./scripts/build_mac.sh
```

Outputs:

- `dist/MotionExtractionUI.app`
- `dist/MotionExtractionUI-mac.zip`

Optional: use a specific local ffmpeg binary instead of auto-download:

```bash
./scripts/build_mac.sh --ffmpeg /absolute/path/to/ffmpeg
```

## GitHub Actions Builds

- macOS app workflow: `.github/workflows/build-macos.yml`

## Usage

Default run (uses `videos/video.mp4` -> `~/Desktop/video_output_f1.mp4`):

```bash
python3 motion_extraction.py
```

Basic (1-frame shift):

```bash
python3 motion_extraction.py input.mp4 output.mp4
```

If you omit `output.mp4`, the file name is auto-generated as:

- Non-RGB: `~/Desktop/<input_stem>_output_f<frame_offset>.mp4`
- RGB split: `~/Desktop/<input_stem>_output_rgb_r<R>_g<G>_b<B>.mp4`
- Freeze-frame: `~/Desktop/<input_stem>_output_freeze<frame>.mp4`
- If `--overlay-original` is enabled, `_overlaid` is appended before `.mp4`.

Choose non-RGB frame offset:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --frame-offset 12
```

(`--frame-offset` is an alias of `--offset-frames`; default remains `1`.)

Shift by seconds:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --offset-seconds 1.0
```

`--offset-seconds` cannot be combined with `--frame-offset` / `--offset-frames`.

Freeze reference frame (show changes relative to one fixed frame):

```bash
python3 motion_extraction.py input.mp4 output.mp4 --freeze-frame 0
```

Enhance larger/stronger motion:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --pre-blur 5 --contrast 2.0
```

Grayscale output:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --grayscale
```

Extract R/G/B channels with different time offsets:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --rgb-offset-frames 1,10,20
```

Use default RGB offsets (`R5,G10,B15`):

```bash
python3 motion_extraction.py input.mp4 output.mp4 --rgb-offset-frames
```

Extract R/G/B channels with per-channel seconds:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --rgb-offset-seconds 0,0.08,0.16
```

`--rgb-offset-frames` is in `R,G,B` order and overrides `--offset-frames` / `--offset-seconds`.
`--rgb-offset-seconds` is also `R,G,B`, converted to frames using the output FPS, and cannot be combined with `--rgb-offset-frames`.

Overlay extracted motion on top of the original video (works with both normal and RGB offsets):

```bash
python3 motion_extraction.py input.mp4 output.mp4 --overlay-original
```

```bash
python3 motion_extraction.py input.mp4 output.mp4 --rgb-offset-frames 1,10,20 --overlay-original
```

Adjust overlay intensity:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --overlay-original --overlay-strength 0.7
```

Adjust color intensity of the motion result:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --saturation 1.5
```

Add temporal persistence trails:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --trail-accumulation-frames 12
```

Trail accumulation by seconds:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --trail-accumulation-seconds 0.4
```

Choose output codec:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --codec avc1
```

Override output FPS:

```bash
python3 motion_extraction.py input.mp4 output.mp4 --fps 30
```

Full example with multiple features:

```bash
python3 motion_extraction.py input.mp4 --rgb-offset-frames 1,8,16 --overlay-original --overlay-strength 0.8 --saturation 1.4 --trail-accumulation-frames 8 --pre-blur 5 --contrast 2.0
```

Notes:

- `--frame-offset` and `--offset-frames` are the same flag.
- `--frame-offset` / `--offset-frames` cannot be combined with `--offset-seconds`.
- `--rgb-offset-frames` overrides `--frame-offset` / `--offset-seconds`.
- `--rgb-offset-seconds` cannot be combined with `--rgb-offset-frames`.
- `--freeze-frame` cannot be combined with `--rgb-offset-frames` or `--rgb-offset-seconds`.
- `--freeze-frame` cannot be combined with `--overlay-original`.
- `--contrast` must be in `[0,10]`.
- `--pre-blur` must be an odd integer in `[0,49]` (or `0` to disable).
- `--overlay-strength` must be in `[0,1]`.
- `--saturation` must be in `[0,3]`.
- `--trail-accumulation-frames` (alias: `--trail-accumulation`) must be `>= 0`.
- `--trail-accumulation-seconds` must be `>= 0`.
- `--trail-accumulation-frames` and `--trail-accumulation-seconds` are mutually exclusive.
- If you pass an explicit output path, that path is used instead of auto naming.
- If Desktop is unavailable, auto-generated output falls back to your home directory.
- Audio is always copied from input to output automatically (video uses OpenCV, audio mux uses `ffmpeg`).
- For source runs, `ffmpeg` must be installed or `MOTION_FFMPEG_PATH` must point to it.

Run tests:

```bash
python3 -m unittest -v
```
