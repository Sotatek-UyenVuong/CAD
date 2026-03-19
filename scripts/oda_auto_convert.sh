#!/bin/bash
# ODA File Converter — auto-click headless conversion script
# Chạy ODA trên xvfb, dùng xdotool tự động click Convert/OK
#
# Usage:
#   ./oda_auto_convert.sh <input_folder> <output_folder>
#   ./oda_auto_convert.sh /path/to/dwg/ /path/to/dxf/
#
# Requires: xvfb-run, xdotool
#   sudo apt install -y xvfb xdotool

set -e

INPUT_DIR="${1:-/tmp/dwg_input}"
OUTPUT_DIR="${2:-/tmp/dwg_output}"
ODA_BIN="${ODA_BIN:-ODAFileConverter}"

# ── Kiểm tra dependencies ──────────────────────────────────────────────────
check_deps() {
    local missing=()
    for cmd in xvfb-run xdotool "$ODA_BIN"; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "❌  Thiếu: ${missing[*]}"
        echo "   sudo apt install -y xvfb xdotool"
        exit 1
    fi
}

# ── Tìm và click nút Convert/OK trong cửa sổ ODA ──────────────────────────
auto_click_oda() {
    local display="$1"
    local max_wait=30
    local waited=0

    echo "  [auto-click] Chờ cửa sổ ODA xuất hiện..."

    while [[ $waited -lt $max_wait ]]; do
        # Tìm window của ODA (tên có thể là "ODA File Converter" hoặc tương tự)
        local wid
        wid=$(DISPLAY="$display" xdotool search --name "ODA" 2>/dev/null | head -1)

        if [[ -n "$wid" ]]; then
            echo "  [auto-click] Tìm thấy window ID: $wid"

            # Đưa window lên focus
            DISPLAY="$display" xdotool windowfocus "$wid"
            sleep 0.5

            # Screenshot để debug (nếu cần)
            # DISPLAY="$display" import -window "$wid" /tmp/oda_screenshot.png 2>/dev/null || true

            # Thử click nút "Convert" (phím Enter thường trigger nút mặc định)
            DISPLAY="$display" xdotool key --window "$wid" Return
            sleep 1

            # Nếu có dialog confirm, nhấn Enter thêm lần nữa
            DISPLAY="$display" xdotool key --window "$wid" Return
            sleep 0.5

            echo "  [auto-click] Đã gửi Enter ✅"
            return 0
        fi

        sleep 1
        ((waited++))
        [[ $((waited % 5)) -eq 0 ]] && echo "  [auto-click] Đã chờ ${waited}s..."
    done

    echo "  [auto-click] ⚠  Timeout: không tìm thấy cửa sổ ODA sau ${max_wait}s"
    # Dump all windows for debugging
    echo "  Các window đang mở:"
    DISPLAY="$display" xdotool search --name "" 2>/dev/null | while read wid; do
        name=$(DISPLAY="$display" xdotool getwindowname "$wid" 2>/dev/null || echo "?")
        echo "    wid=$wid  name='$name'"
    done
    return 1
}

# ── Main ───────────────────────────────────────────────────────────────────
main() {
    check_deps

    mkdir -p "$OUTPUT_DIR"
    echo "Input : $INPUT_DIR"
    echo "Output: $OUTPUT_DIR"
    echo "Files : $(ls "$INPUT_DIR"/*.dwg "$INPUT_DIR"/*.DWG 2>/dev/null | wc -l) DWG file(s)"
    echo ""

    # Chọn DISPLAY trống
    local display_num=200
    while [[ -f "/tmp/.X${display_num}-lock" ]]; do
        ((display_num++))
    done
    local VDISPLAY=":${display_num}"
    echo "Virtual display: $VDISPLAY"

    # Khởi động Xvfb
    Xvfb "$VDISPLAY" -screen 0 1280x1024x24 -nolisten tcp &
    local xvfb_pid=$!
    sleep 1

    trap "kill $xvfb_pid 2>/dev/null; exit" INT TERM EXIT

    # Chạy ODA ở background
    echo "Khởi động ODAFileConverter..."
    DISPLAY="$VDISPLAY" "$ODA_BIN" \
        "$INPUT_DIR" "$OUTPUT_DIR" \
        ACAD2018 ACAD2018 0 1 "*.DWG" DXF &
    local oda_pid=$!

    # Auto-click song song
    auto_click_oda "$VDISPLAY" &
    local click_pid=$!

    # Đợi ODA hoàn thành (tối đa 5 phút)
    local timeout=300
    local elapsed=0
    while kill -0 "$oda_pid" 2>/dev/null && [[ $elapsed -lt $timeout ]]; do
        sleep 2
        ((elapsed+=2))
        # Kiểm tra đã có file output chưa
        local out_count
        out_count=$(ls "$OUTPUT_DIR"/*.dxf "$OUTPUT_DIR"/*.DXF 2>/dev/null | wc -l)
        if [[ $out_count -gt 0 ]]; then
            echo "  ✅  Đã thấy $out_count file DXF trong output"
        fi
    done

    kill "$click_pid" 2>/dev/null || true
    kill "$oda_pid"   2>/dev/null || true
    kill "$xvfb_pid"  2>/dev/null || true
    trap - INT TERM EXIT

    echo ""
    echo "Kết quả:"
    ls -lh "$OUTPUT_DIR"/ 2>/dev/null || echo "Output folder trống"

    local count
    count=$(ls "$OUTPUT_DIR"/*.dxf "$OUTPUT_DIR"/*.DXF 2>/dev/null | wc -l)
    if [[ $count -gt 0 ]]; then
        echo "✅  Convert thành công: $count file DXF"
    else
        echo "❌  Không tìm thấy file DXF — thử chạy manual hoặc kiểm tra log"
        exit 1
    fi
}

main "$@"
