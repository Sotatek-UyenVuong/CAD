#!/bin/bash
# ODA File Converter — headless auto-click
# Usage: bash run_convert.sh <input_folder> <output_folder>
set -e

INPUT="${1:?Usage: $0 <input_folder> <output_folder>}"
OUTPUT="${2:?Usage: $0 <input_folder> <output_folder>}"
mkdir -p "$OUTPUT"

# Find free Xvfb display
DNUM=201
while [[ -f "/tmp/.X${DNUM}-lock" ]]; do ((DNUM++)); done
VDISP=":${DNUM}"

echo "=== ODA File Converter (headless) ==="
echo "Input : $INPUT"
echo "Output: $OUTPUT"
echo "Display: $VDISP"
echo ""

# Start Xvfb
Xvfb "$VDISP" -screen 0 1280x1024x24 -nolisten tcp &
XVFB_PID=$!
sleep 2
trap "kill $XVFB_PID 2>/dev/null" EXIT

# Auto-click OK dialog in background
(
    for i in $(seq 1 20); do
        sleep 1
        WID=$(DISPLAY="$VDISP" xdotool search --name "ODA File Converter" 2>/dev/null | tail -1)
        if [[ -n "$WID" ]]; then
            echo "  [auto-click] Found window after ${i}s (wid=$WID) → clicking OK"
            DISPLAY="$VDISP" xdotool windowfocus --sync "$WID" 2>/dev/null || true
            sleep 0.3
            DISPLAY="$VDISP" xdotool mousemove --window "$WID" 240 280 2>/dev/null || true
            sleep 0.2
            DISPLAY="$VDISP" xdotool click 1 2>/dev/null || true
            break
        fi
    done
) &
CLICK_PID=$!

# Run ODA — correct 7-arg format:
#   <InputFolder> <OutputFolder> <OutputVersion> <OutputType> <Recurse> <Audit> [filter]
echo "Converting..."
DISPLAY="$VDISP" ODAFileConverter \
    "$INPUT" "$OUTPUT" \
    ACAD2018 DXF 0 1 "*.DWG"

kill $CLICK_PID 2>/dev/null || true

echo ""
echo "=== Result ==="
ls -lh "$OUTPUT"/
COUNT=$(ls "$OUTPUT"/*.dxf "$OUTPUT"/*.DXF 2>/dev/null | wc -l)
echo ""
if [[ $COUNT -gt 0 ]]; then
    echo "✅  $COUNT DXF file(s) converted successfully"
else
    echo "❌  No DXF files in output"
    exit 1
fi
