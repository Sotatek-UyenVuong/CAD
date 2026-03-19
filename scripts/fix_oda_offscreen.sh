#!/bin/bash
# Fix ODA File Converter để chạy headless (không cần display)
# Patch libqoffscreen.so dùng đúng Qt libs của ODA
set -e

ODA_DIR="/usr/bin/ODAFileConverter_27.1.0.0"
ODA_PLUGINS="$ODA_DIR/plugins/platforms"
ODA_LIB="$ODA_DIR/lib"
OFFSCREEN_SRC="/usr/lib/x86_64-linux-gnu/qt6/plugins/platforms/libqoffscreen.so"
OFFSCREEN_DST="$ODA_PLUGINS/libqoffscreen.so"

echo "=== Fix ODA Headless (offscreen) ==="

# ── Kiểm tra deps ──────────────────────────────────────────────────────────
for cmd in patchelf ODAFileConverter; do
    command -v "$cmd" &>/dev/null || { echo "❌ Thiếu: $cmd"; echo "  sudo apt install -y patchelf qt6-qpa-plugins"; exit 1; }
done

[[ -f "$OFFSCREEN_SRC" ]] || { echo "❌ Không tìm thấy: $OFFSCREEN_SRC"; echo "  sudo apt install -y qt6-qpa-plugins"; exit 1; }

# ── Copy và patch libqoffscreen.so ─────────────────────────────────────────
echo "1. Copy libqoffscreen.so → $OFFSCREEN_DST"
cp "$OFFSCREEN_SRC" "$OFFSCREEN_DST"
chmod 755 "$OFFSCREEN_DST"

echo "2. Patch rpath → ODA's own Qt ($ODA_LIB)"
patchelf --set-rpath "$ODA_LIB" "$OFFSCREEN_DST"
echo "   RPATH: $(patchelf --print-rpath "$OFFSCREEN_DST")"

# ── Verify ─────────────────────────────────────────────────────────────────
echo "3. Verify dependencies (không còn 'not found'):"
ldd "$OFFSCREEN_DST" | grep "not found" && echo "   ⚠  Vẫn còn missing libs!" || echo "   ✅ Tất cả deps OK"

# ── Test convert ───────────────────────────────────────────────────────────
echo ""
echo "4. Test convert headless..."
rm -rf /tmp/oda_test_out && mkdir -p /tmp/oda_test_in /tmp/oda_test_out

# Tạo file DWG test nhỏ
cp "/home/sotatek/Documents/Uyen/cad/D000_表紙-意匠.dwg" /tmp/oda_test_in/D000.DWG 2>/dev/null || \
  cp /tmp/dwg_input/D000.DWG /tmp/oda_test_in/ 2>/dev/null || \
  { echo "⚠  Không có file DWG test, bỏ qua bước test"; exit 0; }

QT_QPA_PLATFORM=offscreen \
LD_LIBRARY_PATH="$ODA_LIB:$LD_LIBRARY_PATH" \
ODAFileConverter \
  /tmp/oda_test_in /tmp/oda_test_out \
  ACAD2018 ACAD2018 0 1 "*.DWG" DXF 2>&1
EXIT=$?

echo ""
if ls /tmp/oda_test_out/*.dxf /tmp/oda_test_out/*.DXF 2>/dev/null | head -1 | grep -q .; then
    echo "✅ THÀNH CÔNG! File DXF:"
    ls -lh /tmp/oda_test_out/
else
    echo "Exit code: $EXIT"
    echo "❌ Chưa ra file DXF — xem log ở trên"
fi
