"""
CAD Detection Chatbot
Chatbot hỏi đáp để nhận diện các vùng trong bản vẽ CAD.

Luồng:
  User nói muốn detect cái gì
    → LLM gen prompt nhận diện chuyên biệt
    → Gemini Vision detect + trả JSON
    → Vẽ bounding box lên ảnh
    → Lưu kết quả vào chatbot_exports/
"""

import os
import sys
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fitz  # pymupdf
from prompts import build_system_prompt, build_json_suffix

load_dotenv()

# ─────────────────────────────────────────────
# Object catalog từ object_descriptions.json
# ─────────────────────────────────────────────
def _load_object_catalog() -> str:
    """Load object_descriptions.json và build catalog string để inject vào system prompt."""
    catalog_path = Path(__file__).parent / "object_descriptions.json"
    if not catalog_path.exists():
        return ""
    try:
        with open(catalog_path, encoding="utf-8") as f:
            data = json.load(f)
        lines = ["OBJECT CATALOG (use these descriptions when generating detection prompts):"]
        for obj in data.get("objects", []):
            name_en  = obj.get("name_en", "")
            name_ja  = obj.get("name_ja", "")
            name_vi  = obj.get("name_vi", "")
            category = obj.get("category", "")
            desc     = obj.get("description", "")
            hint     = obj.get("shape_hint", "")
            excl     = obj.get("exclude_hints", [])
            entry = (
                f"• [{category}] {name_en} / {name_ja} / {name_vi}\n"
                f"  Visual description: {desc}\n"
                f"  Shape hint: {hint}"
            )
            if excl:
                entry += "\n  Exclude hints: " + "; ".join(excl)
            lines.append(entry)
        return "\n".join(lines)
    except Exception:
        return ""

OBJECT_CATALOG = _load_object_catalog()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Fallback chain: thử lần lượt khi model bị 503 / 429
MODELS_PROMPT = [
    "gemini-2.5-flash",
 
]
MODELS_DETECT = [
    "gemini-3.1-pro-preview",
]

OUTPUT_DIR = "chatbot_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not API_KEY:
    print("❌ Cần đặt GEMINI_API_KEY trong .env")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

_PDF_PAGE_CACHE: dict[tuple[str, int], tuple[float, float, float, float, int]] = {}

# ─────────────────────────────────────────────
# Color palette (RGB)
# ─────────────────────────────────────────────
PALETTE = [
    (220,  20,  60), (  0, 180,   0), (  0,  80, 220), (220, 130,   0),
    (140,   0, 210), (  0, 180, 180), (200,  80,   0), (  0, 150, 100),
    (180,   0, 120), ( 80, 120, 220),
]

def _color_for(class_name: str, seen: dict) -> tuple:
    if class_name not in seen:
        seen[class_name] = PALETTE[len(seen) % len(PALETTE)]
    return seen[class_name]

# ─────────────────────────────────────────────
# Font helper (hỗ trợ tiếng Nhật)
# ─────────────────────────────────────────────
def _load_font(size: int):
    candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

# ─────────────────────────────────────────────
# Normalize coordinates về [0, 1]
# ─────────────────────────────────────────────
def _norm(vals: list) -> list:
    m = max(abs(v) for v in vals) if vals else 0
    if m <= 1.0:
        scale = 1.0
    elif m <= 10.0:
        scale = 10.0
    elif m <= 100.0:
        scale = 100.0
    else:
        scale = 1000.0
    return [v / scale for v in vals]

# ─────────────────────────────────────────────
# Fallback caller — thử từng model khi 503/429
# ─────────────────────────────────────────────
def _call_with_fallback(models: list, **kwargs) -> str:
    last_err = None
    for model in models:
        try:
            resp = client.models.generate_content(model=model, **kwargs)
            print(f"  📡 Model: {model}")
            return resp.text.strip()
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in ["503", "429", "UNAVAILABLE"]) or "quota" in msg.lower():
                print(f"  ⚠️  {model} không khả dụng, thử tiếp...")
                last_err = e
                continue
            raise
    raise RuntimeError(f"Tất cả models đều lỗi. Lỗi cuối: {last_err}")

SYSTEM_PROMPT_GEN = build_system_prompt(OBJECT_CATALOG)

# ─────────────────────────────────────────────
# Step 1 – Gen detection prompt từ yêu cầu user
# ─────────────────────────────────────────────
def generate_detection_prompt(user_request: str) -> tuple[str, str]:
    """
    Returns (full_prompt, class_name).
    LLM trả về:
      Line 1: CLASS: <name>
      Lines 2+: detection description
    """
    print("  🧠 Đang tạo prompt nhận diện...")
    result = _call_with_fallback(
        MODELS_PROMPT,
        contents=user_request,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_GEN,
            temperature=0.3,
        ),
    )

    # Parse CLASS: <name> từ dòng đầu
    lines = result.strip().splitlines()
    class_name = "object"
    if lines and lines[0].upper().startswith("CLASS:"):
        class_name = lines[0].split(":", 1)[1].strip().lower()
        description = "\n".join(lines[1:]).strip()
    else:
        description = result.strip()

    full_prompt = description + build_json_suffix(class_name)
    print(f"\n  📋 Class: [{class_name}]  Prompt:\n  {'─'*50}\n{full_prompt}\n  {'─'*50}")
    return full_prompt, class_name

# ─────────────────────────────────────────────
# Step 2 – Gọi Gemini Vision detect
# ─────────────────────────────────────────────
def _parse_detections(raw: str) -> list:
    """Parse JSON detections từ raw text (hỗ trợ truncated + regex fallback)."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("detections") or []
    except json.JSONDecodeError:
        pass
    pattern = re.compile(
        r'\{\s*"class"\s*:\s*"([^"]+)".*?"x_min"\s*:\s*([\d.]+).*?"y_min"\s*:\s*([\d.]+).*?"x_max"\s*:\s*([\d.]+).*?"y_max"\s*:\s*([\d.]+)',
        re.DOTALL,
    )
    return [
        {"class": m[0], "label": "", "x_min": float(m[1]), "y_min": float(m[2]),
         "x_max": float(m[3]), "y_max": float(m[4])}
        for m in pattern.findall(raw)
    ]


def run_detection(image_path: str, detection_prompt: str) -> list:
    print("  🔍 Đang phân tích bản vẽ...")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    raw = _call_with_fallback(
        MODELS_DETECT,
        contents=[image_part, detection_prompt],
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=16384),
    )
    return _parse_detections(raw)


def _norm_det(d: dict) -> dict | None:
    """Normalize tọa độ bbox về [0, 1]. Trả None nếu bbox không hợp lệ."""
    try:
        vals = [d["x_min"], d["y_min"], d["x_max"], d["y_max"]]
    except KeyError:
        return None
    m = max(abs(v) for v in vals) if vals else 0
    if m == 0:
        return None
    scale = 1.0 if m <= 1.0 else (10.0 if m <= 10.0 else (100.0 if m <= 100.0 else 1000.0))
    d["x_min"] = vals[0] / scale
    d["y_min"] = vals[1] / scale
    d["x_max"] = vals[2] / scale
    d["y_max"] = vals[3] / scale
    return d


def _display_pt_to_raw_pt(px: float, py: float, mW: float, mH: float, rotation: int) -> tuple[float, float]:
    """
    Convert display-point coordinates (top-left origin) to raw PDF coordinates.
    """
    rot = rotation % 360
    if rot == 0:
        return px, mH - py
    if rot == 90:
        return py, mH - px
    if rot == 180:
        return mW - px, py
    if rot == 270:
        return mW - py, px
    return px, py


def _parse_rendered_image_context(image_path: str) -> tuple[str | None, int | None]:
    """
    Try infer (pdf_path, page_index) from rendered image path:
      rendered_<pdf_stem>[_dpiXXX]/page_<idx>.png
    """
    p = Path(image_path)
    m_page = re.match(r"^page_(\d+)\.png$", p.name, flags=re.IGNORECASE)
    if not m_page:
        return None, None
    page_idx = int(m_page.group(1))

    parent = p.parent.name
    if not parent.startswith("rendered_"):
        return None, None
    stem = parent[len("rendered_"):]
    stem = re.sub(r"_dpi\d+$", "", stem)

    candidates = [
        Path("pdf") / f"{stem}.pdf",
        Path(f"{stem}.pdf"),
    ]
    for c in candidates:
        if c.exists():
            return str(c), page_idx

    # Fallback: recursive search by exact filename
    target_name = f"{stem}.pdf"
    for root, _, files in os.walk("."):
        if target_name in files:
            return str(Path(root) / target_name), page_idx
    return None, None


def _get_pdf_page_meta(pdf_path: str, page_idx: int) -> tuple[float, float, float, float, int] | None:
    """
    Return (display_w, display_h, mediabox_w, mediabox_h, rotation) for a PDF page.
    """
    key = (pdf_path, page_idx)
    if key in _PDF_PAGE_CACHE:
        return _PDF_PAGE_CACHE[key]
    try:
        doc = fitz.open(pdf_path)
        if not (0 <= page_idx < len(doc)):
            return None
        page = doc[page_idx]
        meta = (
            float(page.rect.width),
            float(page.rect.height),
            float(page.mediabox.width),
            float(page.mediabox.height),
            int(page.rotation),
        )
        _PDF_PAGE_CACHE[key] = meta
        doc.close()
        return meta
    except Exception:
        return None



# ─────────────────────────────────────────────
# Step 3 – Vẽ bounding box lên ảnh
# ─────────────────────────────────────────────
def draw_detections(image_path: str, detections: list, out_path: str) -> int:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    thickness = max(3, min(w, h) // 400)
    color_map: dict = {}
    count = 0

    for det in detections:
        try:
            raw_vals = (det["x_min"], det["y_min"], det["x_max"], det["y_max"])
            normed = _norm_det({
                "x_min": float(det["x_min"]), "y_min": float(det["y_min"]),
                "x_max": float(det["x_max"]), "y_max": float(det["y_max"]),
                "class": det.get("class", "object"), "label": det.get("label", ""),
            })
            if normed is None:
                print(f"  ⚠️  skip (norm=None): raw={raw_vals}")
                continue
            det = normed
            x_min, y_min, x_max, y_max = det["x_min"], det["y_min"], det["x_max"], det["y_max"]
        except (KeyError, ValueError, TypeError) as e:
            print(f"  ⚠️  skip (parse error {e}): {det}")
            continue

        if x_max <= x_min or y_max <= y_min:
            print(f"  ⚠️  skip (invalid box): x={x_min:.3f}-{x_max:.3f} y={y_min:.3f}-{y_max:.3f}")
            continue

        # Clamp to image bounds
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)

        cls   = str(det.get("class", "object"))
        label = str(det.get("label", cls))
        color = _color_for(cls, color_map)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        # Chỉ vẽ class label (không có tên phòng tiếng Nhật)
        font = _load_font(max(16, min(w, h) // 100))
        bbox = draw.textbbox((0, 0), cls, font=font)
        th   = bbox[3] - bbox[1]
        ty   = max(y1 - th - 2, 0)
        draw.text((x1, ty), cls, font=font, fill=color)
        count += 1

    img.save(out_path)
    return count


def save_detections_json(
    image_path: str,
    detections: list,
    out_path: str,
    user_request: str,
    class_name: str,
    prompt: str,
) -> int:
    """
    Lưu JSON kết quả detect gồm:
      - bbox_norm: tọa độ chuẩn hóa [0,1]
      - bbox_px: tọa độ pixel theo ảnh hiện tại
    """
    img = Image.open(image_path)
    w, h = img.size

    pdf_path, page_idx = _parse_rendered_image_context(image_path)
    page_meta = None
    if pdf_path is not None and page_idx is not None:
        page_meta = _get_pdf_page_meta(pdf_path, page_idx)

    items = []
    for det in detections:
        try:
            normed = _norm_det({
                "x_min": float(det["x_min"]), "y_min": float(det["y_min"]),
                "x_max": float(det["x_max"]), "y_max": float(det["y_max"]),
                "class": det.get("class", "object"),
                "label": det.get("label", ""),
            })
        except (KeyError, ValueError, TypeError):
            continue

        if normed is None:
            continue

        x_min, y_min, x_max, y_max = (
            max(0.0, min(1.0, normed["x_min"])),
            max(0.0, min(1.0, normed["y_min"])),
            max(0.0, min(1.0, normed["x_max"])),
            max(0.0, min(1.0, normed["y_max"])),
        )
        if x_max <= x_min or y_max <= y_min:
            continue

        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)

        det_item = {
            "class": str(normed.get("class", "object")),
            "label": str(normed.get("label", "")),
            "bbox_norm": [round(x_min, 6), round(y_min, 6), round(x_max, 6), round(y_max, 6)],
            "bbox_px": [x1, y1, x2, y2],
        }

        # Add PDF-point coordinates when source image is mapped to PDF page.
        if page_meta is not None:
            disp_w, disp_h, mW, mH, rotation = page_meta
            dx0, dy0 = x_min * disp_w, y_min * disp_h
            dx1, dy1 = x_max * disp_w, y_max * disp_h
            det_item["bbox_pdf_display_pt"] = [
                round(dx0, 3), round(dy0, 3), round(dx1, 3), round(dy1, 3)
            ]

            rx0, ry0 = _display_pt_to_raw_pt(dx0, dy0, mW, mH, rotation)
            rx1, ry1 = _display_pt_to_raw_pt(dx1, dy1, mW, mH, rotation)
            det_item["bbox_pdf_raw_pt"] = [
                round(min(rx0, rx1), 3), round(min(ry0, ry1), 3),
                round(max(rx0, rx1), 3), round(max(ry0, ry1), 3),
            ]

        items.append(det_item)

    payload = {
        "image_path": image_path,
        "image_size": {"width": w, "height": h},
        "user_request": user_request,
        "target_class": class_name,
        "prompt": prompt,
        "count": len(items),
        "detections": items,
    }
    if page_meta is not None and pdf_path is not None and page_idx is not None:
        disp_w, disp_h, mW, mH, rotation = page_meta
        payload["pdf_context"] = {
            "pdf_path": pdf_path,
            "page_index": page_idx,
            "page_rotation": rotation,
            "page_display_size_pt": [round(disp_w, 3), round(disp_h, 3)],
            "page_mediabox_size_pt": [round(mW, 3), round(mH, 3)],
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return len(items)

# ─────────────────────────────────────────────
# Step 4 – In tổng kết
# ─────────────────────────────────────────────
def print_summary(detections: list):
    from collections import Counter
    counts = Counter(str(d.get("class", "?")) for d in detections)
    for cls, n in sorted(counts.items()):
        labels = [str(d.get("label", "")) for d in detections if str(d.get("class")) == cls]
        sample = ", ".join(l for l in labels[:5] if l)
        suffix = f"  ({sample}{'...' if len(labels) > 5 else ''})" if sample else ""
        print(f"    • {cls}: {n} vùng{suffix}")

# ─────────────────────────────────────────────
# Chọn ảnh đầu vào
# ─────────────────────────────────────────────
def pick_image() -> str:
    search_dirs = ["output_png", "rendered_神奈川新町駅", "png","rendered_TLC_BZ商品計画テキスト2013モデルプラン_dpi300"]
    images = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".png"):
                    images.append(os.path.join(d, f))

    if not images:
        return input("📂 Nhập đường dẫn ảnh PNG: ").strip()

    print("\n📂 Ảnh có sẵn:")
    for i, p in enumerate(images, 1):
        print(f"  [{i}] {p}")
    print("  [0] Nhập đường dẫn khác")

    while True:
        choice = input("Chọn số: ").strip()
        if choice == "0":
            return input("Đường dẫn: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(images):
            return images[int(choice) - 1]

# ─────────────────────────────────────────────
# Main chatbot loop
# ─────────────────────────────────────────────
def chatbot():
    print("=" * 60)
    print("  🏗️  CAD DETECTION CHATBOT")
    print("  Nhận diện linh hoạt các vùng trong bản vẽ CAD")
    print("=" * 60)
    print("Lệnh: 'image' đổi ảnh | 'prompt' xem prompt | 'exit' thoát\n")

    image_path = pick_image()
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return

    print(f"\n✅ Đang dùng ảnh: {image_path}")
    print("-" * 60)

    last_prompt = None
    session_count = 0

    while True:
        print()
        user_input = input("🗣️  Bạn muốn detect gì? > ").strip()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 Tạm biệt!")
            break
        if user_input.lower() == "image":
            image_path = pick_image()
            if os.path.exists(image_path):
                print(f"✅ Đã đổi ảnh: {image_path}")
            else:
                print(f"❌ Không tìm thấy: {image_path}")
            continue
        if user_input.lower() == "prompt":
            if last_prompt:
                print(f"\n📋 Prompt lần trước:\n{last_prompt}\n")
            else:
                print("Chưa có prompt nào.")
            continue

        print()

        # Step 1: Gen prompt
        try:
            detection_prompt, class_name = generate_detection_prompt(user_input)
            last_prompt = detection_prompt
        except Exception as e:
            print(f"  ❌ Lỗi khi tạo prompt: {e}")
            continue

        # Step 2: Detect
        try:
            detections = run_detection(image_path, detection_prompt)
        except Exception as e:
            print(f"  ❌ Lỗi khi detect: {e}")
            continue

        if not detections:
            print("  ⚠️  Không tìm thấy vùng nào.")
            continue


        # Step 3: Draw
        session_count += 1
        img_stem = Path(image_path).stem
        safe_req = re.sub(r"[^\w\-]", "_", user_input[:30])
        out_name = f"{img_stem}_{safe_req}_{session_count}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        json_name = f"{img_stem}_{safe_req}_{session_count}.json"
        json_path = os.path.join(OUTPUT_DIR, json_name)

        try:
            n = draw_detections(image_path, detections, out_path)
            n_json = save_detections_json(
                image_path=image_path,
                detections=detections,
                out_path=json_path,
                user_request=user_input,
                class_name=class_name,
                prompt=detection_prompt,
            )
        except Exception as e:
            print(f"  ❌ Lỗi khi vẽ: {e}")
            continue

        # Step 4: Report
        print(f"\n  ✅ Detect xong — {n} vùng được vẽ")
        if n_json != n:
            print(f"  ℹ️  JSON hợp lệ: {n_json} vùng (sau khi lọc bbox lỗi)")
        print_summary(detections)
        print(f"\n  🖼️  Kết quả: {out_path}")
        print(f"  🧾 JSON vị trí: {json_path}")
        print("-" * 60)


if __name__ == "__main__":
    chatbot()
