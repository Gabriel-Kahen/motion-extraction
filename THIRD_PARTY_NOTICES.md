# Third-Party Notices

This project can bundle and/or depend on third-party components.

## ffmpeg

- Project: [FFmpeg](https://ffmpeg.org/)
- Purpose: Audio muxing/copying in output videos.
- Typical source used by `scripts/build_mac.sh`: [evermeet.cx](https://evermeet.cx/ffmpeg/)
- License: FFmpeg is distributed under LGPL/GPL depending on build options.

If you distribute bundled binaries, you are responsible for complying with the
license terms for the specific ffmpeg build you ship.

## Python dependencies

- numpy
- opencv-python
- pyinstaller

Each dependency is subject to its own license. Review installed package
metadata before redistribution.
