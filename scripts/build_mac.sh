#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="MotionExtractionUI"
FFMPEG_URL_DEFAULT="https://evermeet.cx/ffmpeg/getrelease/zip"
VENDOR_DIR="$ROOT_DIR/vendor/ffmpeg-macos"
FFMPEG_VENDOR_PATH="$VENDOR_DIR/ffmpeg"
FFMPEG_BIN="${MOTION_FFMPEG_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ffmpeg)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --ffmpeg"
        exit 1
      fi
      FFMPEG_BIN="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: ./scripts/build_mac.sh [--ffmpeg /path/to/ffmpeg]

Builds MotionExtractionUI.app on macOS with ffmpeg bundled inside.
If --ffmpeg is not provided, a standalone ffmpeg zip is downloaded from evermeet.cx.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$FFMPEG_BIN" ]]; then
  mkdir -p "$VENDOR_DIR"
  if [[ ! -x "$FFMPEG_VENDOR_PATH" ]]; then
    TMP_ZIP="$(mktemp -t motion_ffmpeg_XXXXXX).zip"
    curl -fsSL "$FFMPEG_URL_DEFAULT" -o "$TMP_ZIP"
    unzip -p "$TMP_ZIP" ffmpeg > "$FFMPEG_VENDOR_PATH"
    rm -f "$TMP_ZIP"
    chmod +x "$FFMPEG_VENDOR_PATH"
  fi
  FFMPEG_BIN="$FFMPEG_VENDOR_PATH"
fi

if [[ ! -f "$FFMPEG_BIN" ]]; then
  echo "ffmpeg binary not found: $FFMPEG_BIN"
  exit 1
fi

chmod +x "$FFMPEG_BIN" || true

python3 -m pip install --user pyinstaller >/dev/null

python3 -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --add-binary "$FFMPEG_BIN:." \
  ui.py

ditto -c -k --sequesterRsrc --keepParent \
  "dist/$APP_NAME.app" \
  "dist/$APP_NAME-mac.zip"

echo "Built app: dist/$APP_NAME.app"
echo "Built zip: dist/$APP_NAME-mac.zip"
