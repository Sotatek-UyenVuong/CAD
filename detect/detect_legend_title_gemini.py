"""
Detect Legend & Title positions using Gemini AI
Phát hiện vị trí "bảng chú giải" và "tiêu đề bản vẽ" bằng Gemini
"""

import os
import sys
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file!")
    sys.exit(1)

genai.configure(api_key=api_key)

def detect_legend_and_title(image_path, debug=False, detect_elements=False, rooms_only=False, elevators_only=False):
    """
    Detect legend table and drawing title using Gemini AI
    
    Args:
        image_path: Path to image
        debug: Print debug info
        detect_elements: If True, also detect architectural elements (rooms, elevators, toilets, stairs)
        rooms_only: If True, only detect rooms (focused mode for better accuracy)
    
    Returns:
        dict with detected regions and their bounding boxes
    """
    print(f"🔍 Analyzing: {image_path}")
    
    # Load image
    try:
        pil_image = Image.open(image_path)
        image_width, image_height = pil_image.size
        print(f"   Image size: {image_width}x{image_height}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Prepare prompt for Gemini
    if elevators_only:
        prompt = """Analyze this Japanese CAD/architectural floor plan and detect ONLY the ELEVATORS (エレベーター / EV).

**WHAT TO DETECT - ELEVATORS:**
- Elevator shaft symbols: usually a square or rectangle containing an "X" cross or diagonal lines
- Often labeled "EV", "エレベーター", "エレベータ", or "昇降機"
- May show an elevator car outline (small rectangle) inside the shaft rectangle
- Sometimes grouped as a pair (two elevator shafts side by side)
- Detect each elevator shaft separately

**HOW TO BE PRECISE:**
- The bounding box should tightly wrap the elevator shaft symbol (the square/rectangle with X or car)
- Do NOT include surrounding corridor or lobby area
- Check near stairwells — elevators are often placed next to stairs

**OUTPUT FORMAT - Return ONLY valid JSON:**
{
  "elevators": [
    {"x_min": 48, "y_min": 22, "x_max": 52, "y_max": 30, "label": "EV"},
    {"x_min": 52, "y_min": 22, "x_max": 56, "y_max": 30, "label": "EV2"}
  ]
}

- Coordinates are percentages of image width/height (0-100 scale)
- "label": use the text visible near the shaft (EV, エレベーター, etc.) or leave as "EV" if unlabeled
- Return ONLY the JSON, no other text, no markdown code blocks"""

    elif rooms_only:
        prompt = """Analyze this Japanese CAD/architectural floor plan and detect ONLY the ROOMS (部屋).

Focus exclusively on enclosed room areas that have walls and a visible text label inside or adjacent to them.

**WHAT TO DETECT - ROOMS (部屋):**
- Any enclosed area bounded by walls (実線 solid lines)
- Must have a visible room name label (Japanese text: 会議室, 事務室, 教室, 倉庫, トイレ, etc.)
- Include ALL named rooms: offices, meeting rooms, classrooms, storage rooms, corridors, restrooms, machine rooms
- Each individual room = one bounding box entry
- Bounding box must follow the INNER wall boundaries of each room (wall-to-wall, tight)
- Do NOT include the legend table (凡例), title block (図面名), or open corridors without labels

**HOW TO BE PRECISE:**
- Look at the wall lines (thick black lines) that form each room's perimeter
- Read the room label text printed inside each room
- The box should span from one wall to the opposite wall of that room
- If a large space contains multiple sub-rooms, detect each sub-room separately

**OUTPUT FORMAT - Return ONLY valid JSON:**
{
  "rooms": [
    {"x_min": 20, "y_min": 30, "x_max": 35, "y_max": 50, "label": "会議室"},
    {"x_min": 36, "y_min": 30, "x_max": 50, "y_max": 50, "label": "事務室"}
  ]
}

- Coordinates are percentages of image width/height (0-100 scale)
- "label" must be the exact Japanese text visible in the room
- Return ONLY the JSON, no other text, no markdown code blocks"""

    elif detect_elements:
        prompt = """Analyze this CAD/architectural floor plan and identify ALL instances of the following:

**DOCUMENT METADATA (single instances):**

1. **LEGEND TABLE** (bảng chú giải / 凡例):
   - A TABLE with symbols/icons and their descriptions
   - Usually has header "凡例" or "記号"
   - TIGHT bounding box around table border ONLY

2. **DRAWING TITLE BLOCK** (tiêu đề bản vẽ / 図面名):
   - Title block/cartouche with drawing info
   - Contains drawing name, scale, date, number
   - TIGHT bounding box around title block border ONLY

3. **GENERAL NOTES TABLE** (汎記 / 注記):
   - Numbered notes/instructions table
   - Header "汎記" or "注記" or "備考"
   - TIGHT bounding box around table border ONLY

**ARCHITECTURAL ELEMENTS (multiple instances):**

4. **ROOMS** (phòng / 部屋):
   - Enclosed spaces with walls defining the perimeter
   - Usually labeled with room name/number (会議室, 事務室, etc.)
   - Include ALL rooms in the floor plan
   - Bounding box should encompass the entire room area (wall-to-wall)
   - Examples: offices, meeting rooms, storage, corridors

5. **ELEVATORS** (thang máy / エレベーター):
   - Elevator shaft symbols (usually square/rectangular with "EV" or "エレベーター" label)
   - Often shows elevator car outline inside shaft
   - Multiple instances if multiple elevators
   - Tight box around elevator shaft

6. **TOILETS** (nhà vệ sinh / トイレ / 便所):
   - Bathroom/restroom areas
   - Usually labeled "WC", "トイレ", "便所", or with toilet symbols
   - Contains toilet fixtures (toilets, sinks, urinals)
   - Include both men's and women's restrooms separately
   - Bounding box around entire toilet room area

7. **STAIRS** (cầu thang / 階段):
   - Staircase symbols (parallel lines indicating steps)
   - Usually labeled "階段" or "STAIRS" or has "UP/DN" arrows
   - Multiple instances if multiple staircases
   - Tight box around stair symbol/area

**OUTPUT FORMAT:**

Return ONLY valid JSON in this EXACT format:
```json
{
  "legend": {"x_min": 10, "y_min": 80, "x_max": 30, "y_max": 95} or null,
  "title": {"x_min": 70, "y_min": 85, "x_max": 95, "y_max": 98} or null,
  "notes": {"x_min": 5, "y_min": 5, "x_max": 25, "y_max": 20} or null,
  "rooms": [
    {"x_min": 20, "y_min": 30, "x_max": 40, "y_max": 50, "label": "会議室"},
    {"x_min": 45, "y_min": 30, "x_max": 65, "y_max": 50, "label": "事務室"}
  ],
  "elevators": [
    {"x_min": 50, "y_min": 20, "x_max": 55, "y_max": 25}
  ],
  "toilets": [
    {"x_min": 10, "y_min": 40, "x_max": 18, "y_max": 50, "label": "男子トイレ"},
    {"x_min": 10, "y_min": 52, "x_max": 18, "y_max": 62, "label": "女子トイレ"}
  ],
  "stairs": [
    {"x_min": 60, "y_min": 45, "x_max": 70, "y_max": 55}
  ]
}
```

- Coordinates are percentages (0-100)
- If element not found, use empty array [] or null
- Include "label" for rooms/toilets if text is visible
- Return ONLY the JSON, no other text"""
    else:
        prompt = """Analyze this CAD/architectural drawing and identify THREE specific regions:

1. **LEGEND TABLE** (bảng chú giải / 凡例):
   - A TABLE with symbols/icons in one column and their descriptions/names in other columns
   - Usually has header text "凡例" or "記号" (symbols)
   - Has a rectangular border/frame around the table
   - IMPORTANT: Detect ONLY the table itself, NOT the surrounding drawing elements
   - IMPORTANT: The bounding box should be TIGHT around the table border only
   - Typically located at bottom-left or left side
   - Example: A table with rows showing circles, triangles, text labels and their meanings

2. **DRAWING TITLE BLOCK** (tiêu đề bản vẽ / 図面名):
   - A rectangular title block/cartouche at the bottom or bottom-right
   - Contains drawing name (図名), scale (縮尺), date (年月日), drawing number (図番)
   - Usually has multiple cells/sections with project info, company name, signatures
   - IMPORTANT: Detect ONLY the title block frame, NOT surrounding drawing
   - IMPORTANT: The bounding box should be TIGHT around the title block border only

3. **GENERAL NOTES TABLE** (bảng ghi chú / 汎記 or 注記):
   - A table with numbered notes/instructions (1, 2, 3, 4, 5...)
   - Usually has header text "汎記" or "注記" or "備考" (notes)
   - Contains construction/installation notes, cable specifications, technical instructions
   - Has a rectangular border/frame around the table
   - IMPORTANT: Detect ONLY the notes table itself, NOT surrounding elements
   - IMPORTANT: The bounding box should be TIGHT around the table border only
   - Typically located at top-left, left side, or near the title block

For EACH region you find, provide:
- The bounding box coordinates as percentages (0-100)
- **Be PRECISE**: Draw a tight box around ONLY the table/block, excluding any surrounding drawing elements
- Format: {"x_min": %, "y_min": %, "x_max": %, "y_max": %}

Return your response in this EXACT JSON format:
{
  "legend": {"x_min": 10, "y_min": 80, "x_max": 30, "y_max": 95} or null,
  "title": {"x_min": 70, "y_min": 85, "x_max": 95, "y_max": 98} or null,
  "notes": {"x_min": 5, "y_min": 5, "x_max": 25, "y_max": 20} or null
}

If you cannot find a region, set it to null.
Return ONLY the JSON, no other text."""
    
    try:
        # Create model (use latest Gemini for vision tasks)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        # Generate response
        print("   Calling Gemini API...")
        response = model.generate_content([prompt, pil_image])
        
        if debug:
            print(f"   Raw response: {response.text}")
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Convert percentage coordinates to normalized (0-1)
        # Gemini may return values on different scales: 0-1, 0-10, 0-100, or 0-1000
        def _norm(raw_vals):
            """Detect scale and normalize a list of raw coordinate values to [0, 1]."""
            m = max(abs(v) for v in raw_vals) if raw_vals else 0
            if m <= 1.0:
                scale = 1.0
            elif m <= 10.0:
                scale = 10.0
            elif m <= 100.0:
                scale = 100.0
            else:
                scale = 1000.0
            return [v / scale for v in raw_vals]

        detections = {}

        def _parse_box(raw):
            """Parse a single bounding box dict → normalized coords + YOLO center format."""
            x_min, y_min, x_max, y_max = _norm([
                float(raw['x_min']), float(raw['y_min']),
                float(raw['x_max']), float(raw['y_max'])
            ])
            return {
                'x': (x_min + x_max) / 2.0,
                'y': (y_min + y_max) / 2.0,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            }

        if elevators_only or rooms_only:
            # Focused single-class mode
            for region_type in ['legend', 'title', 'notes']:
                detections[region_type] = None
            key = 'elevators' if elevators_only else 'rooms'
            elements = result.get(key, [])
            if elements and isinstance(elements, list):
                detections[key] = []
                for elem in elements:
                    box = _parse_box(elem)
                    if 'label' in elem:
                        box['label'] = elem['label']
                    detections[key].append(box)
                print(f"   ✅ Found {len(detections[key])} {key}")
            else:
                detections[key] = []
                print(f"   ❌ No {key} found")
        else:
            # Process single-instance document metadata
            for region_type in ['legend', 'title', 'notes']:
                region = result.get(region_type)
                if region and isinstance(region, dict):
                    detections[region_type] = _parse_box(region)
                    d = detections[region_type]
                    print(f"   ✅ Found {region_type}: ({d['x_min']:.3f}, {d['y_min']:.3f}) → ({d['x_max']:.3f}, {d['y_max']:.3f})")
                else:
                    detections[region_type] = None
                    print(f"   ❌ {region_type} not found")

        # Process multi-instance architectural elements (if requested, not rooms_only)
        if detect_elements and not rooms_only:
            for element_type in ['rooms', 'elevators', 'toilets', 'stairs']:
                elements = result.get(element_type, [])
                if elements and isinstance(elements, list):
                    detections[element_type] = []
                    for elem in elements:
                        box = _parse_box(elem)
                        if 'label' in elem:
                            box['label'] = elem['label']
                        detections[element_type].append(box)
                    print(f"   ✅ Found {len(detections[element_type])} {element_type}")
                else:
                    detections[element_type] = []
                    print(f"   ❌ No {element_type} found")
        
        return detections
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"   Response: {response_text}")
        return None
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return None

def save_yolo_format(detections, output_txt_path, classes={'legend': 0, 'title': 1, 'notes': 2, 'room': 3, 'elevator': 4, 'toilet': 5, 'stair': 6}):
    """
    Save detections in YOLO format:
    class_id x_center y_center width height
    """
    with open(output_txt_path, 'w') as f:
        for region_type, data in detections.items():
            # Handle single-instance detections (legend, title, notes)
            if region_type in ['legend', 'title', 'notes']:
                if data is not None:
                    class_id = classes.get(region_type, 0)
                    line = f"{class_id} {data['x']:.6f} {data['y']:.6f} {data['width']:.6f} {data['height']:.6f}\n"
                    f.write(line)
            
            # Handle multi-instance detections (rooms, elevators, toilets, stairs)
            elif region_type in ['rooms', 'elevators', 'toilets', 'stairs']:
                if data and isinstance(data, list):
                    # Map plural to singular for class lookup
                    singular = region_type[:-1]  # rooms -> room, stairs -> stair
                    class_id = classes.get(singular, 0)
                    
                    for box in data:
                        line = f"{class_id} {box['x']:.6f} {box['y']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
                        f.write(line)
    
    print(f"📄 Saved: {output_txt_path}")

def _find_unicode_font(size: int):
    """Return a PIL ImageFont that supports Japanese/Unicode text."""
    candidates = [
        # macOS system fonts with CJK support
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        # Linux fallbacks
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_and_save(image_path, detections, output_dir):
    """
    Draw bounding boxes on the original image and save a single annotated image.
    Uses Pillow for text rendering so Japanese/Unicode labels display correctly.
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"❌ Cannot load image for drawing: {image_path}")
        return

    h, w = image_cv.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Color palette per region type (RGB for Pillow) — vivid, high-contrast
    COLORS = {
        'legend':    (0,   210,   0),
        'title':     (220,   0,   0),
        'notes':     (255, 140,   0),
        'rooms':     (180,   0, 220),
        'elevators': (0,   200, 220),
        'toilets':   (230, 100,   0),
        'stairs':    (100,   0, 255),
    }
    thickness  = max(4, min(w, h) // 300)
    font_size  = max(14, min(w, h) // 120)
    font       = _find_unicode_font(font_size)

    # Convert OpenCV BGR → PIL RGB for drawing
    pil_img = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    def draw_box(x1, y1, x2, y2, color, label):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        # Label text only, no background
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        ty = max(y1 - th - 2, 0)
        draw.text((x1, ty), label, font=font, fill=color)

    for region_type, data in detections.items():
        color = COLORS.get(region_type, (0, 200, 0))

        if region_type in ['legend', 'title', 'notes']:
            if data is None:
                continue
            x1 = int(data['x_min'] * w)
            y1 = int(data['y_min'] * h)
            x2 = int(data['x_max'] * w)
            y2 = int(data['y_max'] * h)
            draw_box(x1, y1, x2, y2, color, region_type)
            print(f"🖊️  Drew {region_type}: ({x1},{y1})-({x2},{y2})")

        elif region_type in ['rooms', 'elevators', 'toilets', 'stairs']:
            if not data or not isinstance(data, list):
                continue
            for idx, box in enumerate(data, 1):
                x1 = int(box['x_min'] * w)
                y1 = int(box['y_min'] * h)
                x2 = int(box['x_max'] * w)
                y2 = int(box['y_max'] * h)
                lbl = box.get('label', str(idx))
                draw_box(x1, y1, x2, y2, color, f"{region_type} {lbl}")
                print(f"🖊️  Drew {region_type} {idx} ({lbl}): ({x1},{y1})-({x2},{y2})")

    output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
    pil_img.save(output_path)
    print(f"🖼️  Saved annotated image: {output_path}")

def process_images(input_dir, output_dir, crop_dir=None, classes_file=None,
                   detect_elements=False, rooms_only=False, elevators_only=False):
    """
    Process all PNG images in a directory
    
    Args:
        input_dir: Input directory with PNG images
        output_dir: Output directory for YOLO txt files
        crop_dir: Optional directory for cropped images
        classes_file: Optional classes.txt file to create
        detect_elements: If True, detect all architectural elements
        rooms_only: If True, detect only rooms (focused mode)
    """
    print("="*60)
    if elevators_only:
        print("DETECT ELEVATORS ONLY WITH GEMINI (focused mode)")
    elif rooms_only:
        print("DETECT ROOMS ONLY WITH GEMINI (focused mode)")
    elif detect_elements:
        print("DETECT LEGEND, TITLE & ARCHITECTURAL ELEMENTS WITH GEMINI")
    else:
        print("DETECT LEGEND & TITLE WITH GEMINI")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if crop_dir:
        print(f"Crop directory: {crop_dir}")
    if elevators_only:
        print("Mode: Elevators only (tập trung detect thang máy)")
    elif rooms_only:
        print("Mode: Rooms only (tập trung detect phòng)")
    elif detect_elements:
        print("Mode: Detect architectural elements (rooms, elevators, toilets, stairs)")
    print()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if crop_dir:
        os.makedirs(crop_dir, exist_ok=True)

    # Get all PNG files
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    if len(image_files) == 0:
        print("❌ No PNG files found!")
        return

    print(f"Found {len(image_files)} images")
    print()

    # Class mapping
    if elevators_only:
        classes = {'elevator': 0}
    elif rooms_only:
        classes = {'room': 0}
    elif detect_elements:
        classes = {'legend': 0, 'title': 1, 'notes': 2, 'room': 3, 'elevator': 4, 'toilet': 5, 'stair': 6}
    else:
        classes = {'legend': 0, 'title': 1, 'notes': 2}

    # Save classes.txt
    if classes_file:
        with open(classes_file, 'w') as f:
            for name in classes:
                f.write(f"{name}\n")
        print(f"📄 Saved classes: {classes_file}")
        print()

    # Process each image
    success_count = 0

    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_file}...")

        image_path = os.path.join(input_dir, image_file)
        base_name = os.path.splitext(image_file)[0]

        # Detect with Gemini
        detections = detect_legend_and_title(
            image_path,
            detect_elements=detect_elements,
            rooms_only=rooms_only,
            elevators_only=elevators_only,
        )
        
        if detections is None:
            print(f"   ⚠️  Skipping {image_file} (detection failed)")
            print()
            continue
        
        # Check if at least one region was found
        has_metadata = (detections.get('legend') is not None or
                        detections.get('title') is not None or
                        detections.get('notes') is not None)
        has_rooms = len(detections.get('rooms', [])) > 0
        has_elements = (has_rooms or
                        len(detections.get('elevators', [])) > 0 or
                        len(detections.get('toilets', [])) > 0 or
                        len(detections.get('stairs', [])) > 0)

        if not has_metadata and not has_elements:
            print(f"   ⚠️  No regions/elements detected in {image_file}")
            print()
            continue
        
        # Save YOLO format txt
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        save_yolo_format(detections, txt_path, classes)
        
        # Draw bounding boxes and save annotated image
        if crop_dir:
            draw_and_save(image_path, detections, crop_dir)
        
        success_count += 1
        print()
    
    print("="*60)
    print(f"✅ Completed: {success_count}/{len(image_files)} images")
    print("="*60)

def main():
    if len(sys.argv) < 2:
        print("Detect Legend, Title & Architectural Elements using Gemini AI")
        print("\nUsage:")
        print("  python detect_legend_title_gemini.py <input_dir> [options]")
        print("\nExamples:")
        print("  # Basic - chỉ detect legend, title, notes")
        print("  python detect_legend_title_gemini.py output_png")
        print()
        print("  # Detect cả architectural elements")
        print("  python detect_legend_title_gemini.py output_png --elements")
        print()
        print("  # Chỉ detect rooms (focused mode - thường chính xác hơn)")
        print("  python detect_legend_title_gemini.py output_png --rooms-only --output labels --crop cropped_regions --classes classes.txt")
        print()
        print("Options:")
        print("  --output DIR:   Output directory for YOLO txt files (default: input_dir)")
        print("  --crop DIR:     Save annotated images to directory")
        print("  --classes FILE: Save classes.txt file")
        print("  --elements:        Detect all architectural elements")
        print("  --rooms-only:      Detect ONLY rooms (focused, usually more accurate)")
        print("  --elevators-only:  Detect ONLY elevators (focused mode)")
        sys.exit(1)

    input_dir = sys.argv[1]

    # Parse options
    output_dir = input_dir
    crop_dir = None
    classes_file = None
    detect_elements = False
    rooms_only = False
    elevators_only = False

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--crop' and i + 1 < len(sys.argv):
            crop_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--classes' and i + 1 < len(sys.argv):
            classes_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--elements':
            detect_elements = True
            i += 1
        elif sys.argv[i] == '--rooms-only':
            rooms_only = True
            i += 1
        elif sys.argv[i] == '--elevators-only':
            elevators_only = True
            i += 1
        else:
            i += 1
    
    # Check input directory exists
    if not os.path.isdir(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Process images
    process_images(input_dir, output_dir, crop_dir, classes_file, detect_elements, rooms_only, elevators_only)

if __name__ == '__main__':
    main()
