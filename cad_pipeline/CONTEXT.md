# CAD Pipeline — Context for new chat

## Mục tiêu project
Sub-project độc lập tại `/mnt/data8tb/notex/uyenvuong/CAD/cad_pipeline/`  
Index bản vẽ CAD (PDF/DXF) → Q&A + semantic search bằng AI.

---

## Cấu trúc file

```
cad_pipeline/
├── config.py                   # Tất cả config đọc từ .env
├── .env.example                # Template env
├── requirements.txt
├── CONTEXT.md                  # File này
│
├── core/
│   ├── pdf_to_images.py        # PDF → PNG pages (PyMuPDF, 300 DPI)
│   ├── layout_detect.py        # Detectron2 model wrapper (text/table/diagram/title_block/image)
│   ├── block_sorter.py         # Reading-order sort + group (column-major, merge adjacent text/table)
│   ├── page_processor.py       # sort+group → merged crop → Marker/Gemini per group (concurrent)
│   ├── marker_pdf.py           # Chunked PDF → Marker (large docs >20 pages, page remapping)
│   ├── context_builder.py      # Build Markdown context_md per page (blocks pre-sorted)
│   └── embeddings.py           # Cohere embed-multilingual-v3.0 (1024-dim)
│
├── storage/
│   ├── mongo.py                # MongoDB CRUD: folders / files / pages / chat_history
│   ├── qdrant_store.py         # Qdrant vector CRUD (upsert/search/delete)
│   └── s3_store.py             # Cloudflare R2 upload (boto3, no ACL)
│
├── agents/
│   ├── router.py               # Classify query: "qa" | "search"
│   ├── folder_agent.py         # Level 1: answer từ folder/file summary
│   ├── file_agent.py           # Level 2: answer từ file summary
│   └── page_agent.py           # Core: Gemini Pro reasoning + auto-call tools
│
├── tools/
│   ├── count_tool.py           # Đếm symbol (DXF exact → Gemini Pro Vision fallback)
│   ├── viz_tool.py             # Vẽ bounding box lên ảnh (DXF WCS coords / Vision % coords)
│   └── area_tool.py            # Tính diện tích (unit_room_catalog → LLM fallback)
│
├── object_descriptions.json    # 94 object visual descriptions (shape_hint, exclude_hints, name_ja)
│
├── pipeline/
│   ├── upload_pipeline.py      # Full upload flow (9 steps)
│   ├── delete_pipeline.py      # Xóa file/folder: Qdrant + MongoDB + rebuild summary
│   ├── qa_pipeline.py          # Chatbot: Router→Folder→File→Page→Tool + chat history
│   └── search_pipeline.py      # Semantic search: embed→Qdrant→top-N (độc lập với chatbot)
│
└── api/
    └── app.py                  # FastAPI server port 8001
```

---

## Config (.env keys)

```env
# MongoDB
DATABASE_URL=mongodb+srv://...
DATABASE_NAME=cad_pipeline

# Qdrant — local Docker port 6340 (data: /mnt/data8tb/.../qdrant_storage)
QDRANT_URL=http://localhost:6340
QDRANT_API_KEY=None
QDRANT_COLLECTION=cad_pages

# Marker API (Datalab.to) — OCR cho table blocks
MARKER_API_KEY=...

# Gemini
GEMINI_API_KEY=...
# GEMINI_FLASH_MODEL = gemini-2.5-flash       (text, table fallback, title_block, summary)
# GEMINI_PRO_MODEL   = gemini-3.1-pro-preview (diagram, image)

# Cohere (embed, không dùng rerank)
COHERE_API_KEY=...
EMBEDDING_MODEL=embed-multilingual-v3.0

# Search tuning
TOP_K=100
TOP_N=15
SIMILARITY_CUTOFF_VECTORSEARCH=0.2

# Storage — tắt R2, lưu local
USE_S3=false
IMAGES_DIR=./data/images               # local temp dir cho page PNGs

# (Giữ R2 keys phòng khi bật lại sau)
ACCOUNT_ID=...
CLIENT_ACCESS_KEY=...
CLIENT_SECRET=...
R2_BUCKET_NAME=...
R2_PUBLIC_URL=...

# Pipeline
PDF_DPI=300
LAYOUT_SCORE_THR=0.5
AGENT_MAX_PAGES=25
```

---

## Các quyết định thiết kế quan trọng

### 1. OCR / Vision strategy (page_processor.py)

#### Per-block type (solo block)
| Block type | Tool | Ghi chú |
|------------|------|---------|
| `text` | Gemini 2.5 Flash vision | Transcribe chính xác |
| `table` | Marker API (fast mode) → fallback Gemini 2.5 Flash | Nếu Marker trả empty/error → Gemini Flash vision |
| `diagram` | Gemini 3.1 Pro Preview vision | Mô tả chi tiết |
| `image` | Gemini 3.1 Pro Preview vision | Giống diagram (5th class của model) |
| `title_block` | Gemini 3.1 Pro Preview vision | Đọc trực tiếp từ ảnh → JSON, **không qua OCR** |
| Page summary | Gemini 2.5 Flash vision | 1-2 câu tổng quan |

#### Block sort + group (core/block_sorter.py) — chạy TRƯỚC OCR
```
Detect blocks
    ↓ sort_reading_order()
        - Phát hiện cột: x-center gap analysis (threshold = max(50px, median_gap × 1.5))
        - Full-width blocks (span > 75% page width) → tách riêng, luôn xếp CUỐI
        - Trong mỗi cột: sort bằng (y_bucket=50px, x1) → cùng hàng thì trái→phải
        - Cột sort trái→right bằng min x1
    ↓ group_text_table_runs()
        - Gộp blocks text/table liên tiếp cùng cột + cùng _col_id + không cùng hàng (y_gap ≤ 60px)
        - Không gộp nếu cùng y_bucket (side-by-side)
    ↓ Với mỗi group:
        - Solo (1 block)   → crop block bbox → OCR/Gemini như cũ
        - Merged (≥2 text/table) → merge tất cả bbox → 1 crop lớn → 1 Marker call
              (giảm số API calls: 12 tables page 4 → 4 Marker calls)
          Fallback: Marker empty/error → Gemini 2.5 Flash
    ↓ Tất cả groups xử lý song song (ThreadPoolExecutor max_workers=6)
```

**Ví dụ page 4 (13 blocks → 6 groups → 4 Marker calls):**
| Group | Blocks | Crop | Tool |
|-------|--------|------|------|
| G1 | table×3 (col 0) | 609×1545px | Marker × 1 |
| G2 | table×2 (col 1) | 593×1645px | Marker × 1 |
| G3 | table×3 (col 2) | 603×1518px | Marker × 1 |
| G4 | text (col 3) | 701×38px | Gemini Flash |
| G5 | table×3 (col 4) | 619×1522px | Marker × 1 |
| G6 | title_block (full-width) | 2403×89px | Gemini Pro |

**Large PDF optimization (`core/marker_pdf.py`):**
- PDF > 20 trang → chạy `marker_ocr_pdf()` **một lần** trước per-page loop (Step 2b)
- Split thành chunks 10 trang, submit tất cả chunks song song lên Marker
- Marker trả page index **0-based trong chunk** → remap về **1-based original page number**
  - Chunk bắt đầu từ page 0 → offset=0 → pages 1..10
  - Chunk bắt đầu từ page 10 → offset=10 → pages 11..20
- Kết quả cache `{page_number: markdown}` → merged group dùng luôn, skip Marker crop call
- PDF ≤ 20 trang → skip, dùng per-crop/per-group Marker như trên

### 2. Embedding
- Provider: **Cohere** `embed-multilingual-v3.0` (1024-dim, hỗ trợ Japanese+English)
- Upload: `input_type="search_document"`
- Query: `input_type="search_query"` (dùng `embed_query()`)
- **Không dùng Rerank** — chỉ vector search thuần

### 3. Storage strategy (USE_S3=false → local)
- `USE_S3=false` → images lưu local tại `IMAGES_DIR=./data/images`
- Path structure: `data/images/{file_id}/pages/page_{n}.png`
- `image_url` lưu trong MongoDB là **local path** hoặc **HTTP URL** tùy cách serve
- **Serve ảnh qua web:** FastAPI mount `StaticFiles` → trỏ domain → ảnh hiện bình thường (xem mục bên dưới)
- `USE_S3=true` vẫn supported để switch sang R2 khi cần (giữ keys trong .env)

### 4. Symbol DB lookup — Group-based flow

```
User query (any language: "cầu thang" / "階段" / "staircase")
  ↓
Gemini Flash chọn group(s) từ 21 nhóm  ← prompt rất nhỏ, nhanh
  "cầu thang" → ["stair_ramp"]
  "thang máy" → ["elevator_escalator"]
  "fire exit" → ["fire_safety"]
  ↓ (fallback: keyword substring match trên group.keywords nếu Gemini lỗi)
  ↓
symbol_groups.json: group → danh sách labels
  "stair_ramp" → [stair, stair_core, stair_L_shape, stair_railing, ...] (21 labels)
  ↓
symbols.json: label → block_names  (tra cứu tĩnh, không cần LLM)
  "stair_core" → ["コア9F-21F", "コア4F-8F", ...]
  "stair"      → ["避難階段(3)_2F以上", ...]
  ↓
block_names = union tất cả
  ↓
ezdxf: đếm INSERT entity theo block_names  ← đây là đếm thực
```

**Tại sao dùng group làm tầng trung gian:**
- 21 nhóm << 635 labels → prompt Gemini nhỏ hơn 30x, nhanh hơn, chính xác hơn
- Group đã có keywords đa ngôn ngữ (JP/EN/vi) → Gemini dễ match
- Từ group → labels → block_names là tra cứu tĩnh, không cần thêm LLM call nào

### 5. Count tool logic
```
User hỏi "có bao nhiêu EV / elevator?"
  ↓
Symbol DB lookup → block_names = {"EV_*", "A$C123...", ...}
  ↓
  ├── CÓ DXF:
  │     1a. INSERT entity khớp block_names từ symbol DB    → dxf_exact (100%)
  │         + bổ sung TEXT label positions (merge để viz đầy đủ)
  │         → count = max(INSERT count, unique TEXT labels)
  │     1b. Nếu 1a = 0 → fallback DXF:
  │           i.  Scan block definitions: tìm block nào chứa text khớp query
  │               → đếm INSERT của composite block đó     → dxf_symbol_dict
  │           ii. Nếu vẫn 0 → đếm TEXT entity trong modelspace khớp query
  │               (dùng cho sàn B1F: EV1/EV2 là TEXT label riêng lẻ, không phải block)
  │               → Gemini Flash filter labels (loại hall/room/annotation)
  │                                                        → dxf_text_label
  │     Kết quả: {"count", "positions": [{wcs_x, wcs_y, label, block_name}]}
  │
  └── Chỉ PDF/ảnh → Gemini Pro Vision đọc trực tiếp ảnh   → vision_pro
        · Symbol DB lookup → related labels hint
        · Object description lookup (Gemini Flash → object_descriptions.json)
            → description, shape_hint, exclude_hints inject vào prompt
        · Kết quả: {"count", "positions": [{x_min%, y_min%, x_max%, y_max%}]}
```

**Tại sao cần 3 chế độ DXF:**
- Tầng chuẩn (3F–29F): dùng composite block `コア3F`, `コア9F-21F`... → mode 1b-i
- Tầng đặc biệt (B1F, 1F): vẽ thủ công từng thang → chỉ có TEXT "EV1","EV2"... → mode 1b-ii
- Symbol DB khớp chính xác → mode 1a nhanh nhất

- **KHÔNG dùng `count` field trong symbols_enriched.json** — field đó là tần suất trong training data

**Object description lookup (dùng trong Vision mode):**
```
query (any language)
  ↓
Gemini Flash ← compact index JSON
  {category: [{id, vi, en, ja}]}   (94 objects, build 1 lần cache)
  ↓  ["elevator"]
id → full object từ object_descriptions.json
  ↓
description + shape_hint + exclude_hints inject vào Vision prompt
```
- 94 objects, 11 categories: door, window, stair, equipment, furniture,
  home_appliance, lighting, room_label, storage, utility_shaft, annotation
- Gemini Flash match semantic đa ngôn ngữ (vi/en/ja) → chính xác hơn lexical
- Vision hiểu được visual pattern cụ thể (e.g. "hình bầu dục + két nước áp tường")
  → tránh false positive / false negative khi đếm

**Visualization (`viz_tool.py`):**
```
count_result (có "positions")
  ├── mode dxf_*: WCS coords → ViewportTransform → pixel
  │     · get_viewport_bounds_from_dxf() lấy bounds của viewport lớn nhất
  │     · wcs_to_px: linear map WCS → image pixel (y-flip)
  └── mode vision_pro: x_min%/x_max%/y_min%/y_max% → scale pixel
        · _pct_to_px() xử lý cả 0-1 float và 0-100 percent
  → OpenCV: semi-transparent filled rect + border
  → Pillow: Unicode text (NotoSansCJK / DejaVuSans) cho label + header
```

### 5. Area tool logic
```
  ├── có unit_label → tra unit_room_catalog.json (tatami → m²)
  ├── có image_path  → Gemini Pro Vision extract trực tiếp từ ảnh upload
  └── còn lại        → Gemini Flash extract từ context_md
```

### 6. Upload pipeline (9 bước)
```
[1] USE_S3=true  → upload original → R2
    USE_S3=false → skip, dùng local path
[2] Render PDF → page PNGs (local: data/images/{file_id}/pages/)
[2b] PDF > 20 trang → marker_ocr_pdf() — split 10-page chunks, submit song song
     → remap page index (chunk-local 0-based → original 1-based)
     → cache {page_number: markdown} dùng cho bước [4]
     PDF ≤ 20 trang → skip, dùng per-crop Marker như cũ
[3] Tạo folder/file record MongoDB
[4] Per page (xử lý tuần tự từng page):
    → USE_S3=true: upload page PNG → R2 | false: lưu local path
    → Layout detection (Detectron2 R_101_FPN_3x, 5 classes)
      · Nếu thiếu Detectron2: auto degrade (blocks=[]), không fail toàn bộ upload
    → [PARALLEL] Gemini page summary  ┐ chạy đồng thời
                 Process blocks        ┘ (ThreadPoolExecutor)
         └─ process_page_blocks():
              1. sort_reading_order() → column-major reading order
              2. group_text_table_runs() → gộp text/table liên tiếp cùng cột
              3. Per group (song song, max 6):
                 - Solo → crop block → OCR/Gemini
                 - Merged → merge bbox → 1 crop → 1 Marker call (fallback Flash)
              (cache [2b] nếu có → skip Marker, dùng precomputed markdown)
    → Build context_md (blocks đã sorted, render thẳng theo thứ tự)
    → Save page → MongoDB
    → Cohere embed (search_document)
[8] Batch upsert vectors → Qdrant local (port 6340)
[9] Build & save summaries:
    → build_file_summary() — full summary (tất cả pages) → files.summary
    → generate_file_short_summary() — Gemini 2.5 Flash đọc ≤10 page summaries
         → viết 2-3 câu văn xuôi tiếng Nhật → files.short_summary
    → build_folder_summary() — ghép short_summary tất cả files trong folder
         → folders.summary (rebuild mỗi lần có file mới)
    → Build title-block index per file:
         · duyệt processed_blocks, lấy block type=title_block
         · chuẩn hóa metadata (drawing_no, drawing_title, project) theo page
         · lưu vào files.title_block_index để lookup nhanh ảnh→file/page
```

**Runtime resiliency (2026-04-22):**
- `cv2.imread` không còn là single point of failure trong upload path.
  - Nếu `cv2.imread` không tồn tại hoặc đọc lỗi, pipeline fallback sang Pillow (`PIL`) để load ảnh rồi convert về BGR numpy.
- Khi Detectron2 không khả dụng:
  - upload vẫn tiếp tục với `blocks=[]` (không có layout boxes),
  - vẫn tạo page summary, context, embeddings, Mongo/Qdrant records.
- Mục tiêu: ưu tiên ingest thành công thay vì fail cứng do thiếu dependency cục bộ.

**Tại sao cần 2 tầng file summary?**
- `files.summary` (full) → dùng khi File Agent cần tra cứu chi tiết từng page
- `files.short_summary` (Gemini-generated) → nhét vào `folders.summary` để Folder Agent đọc được overview tất cả files mà không bị quá dài

### 7. Chatbot pipeline (Q&A)
```
User query + session_id (khuyến nghị) / folder_id (backward-compatible)
  → Load chat history (5 turns gần nhất từ MongoDB chat_history)
  → Nếu có session_id:
      lấy session_file_ids từ chat_sessions.file_ids làm scope chuẩn
      (cho phép file khác folder trong cùng 1 session)
  → Router (Gemini 2.5 Flash): keyword match → "qa" | "search"
      qa:
        → Direct-chat quick route (Gemini Flash, trước orchestrator):
            · Áp dụng cho chào hỏi / small-talk / câu hỏi ngoài phạm vi tài liệu CAD
            · Trả lời trực tiếp theo ngôn ngữ user
            · Không chạy plan/tool/page pipeline
        → Orchestrator Agent (Gemini 3.1 Pro) lập execution plan cho từng turn:
            · Input: query + chat_history + has_image + tool_router_hint
            · Output plan:
                - grounding_mode: uploaded_image | page_context | hybrid
                - preferred_tool
                - use_history_shortcut
                - force_page_level
                - allow_folder_direct_answer / allow_file_direct_answer
                - allow_image_early_answer
            · Có early-exit: dừng sớm khi evidence đủ mạnh (không bắt buộc chạy hết pipeline)
            · Có replan sau từng bước "trả lời sớm" (history/image/folder/file):
                - orchestrator đánh giá provisional answer + tool_result
                - trả `next_tool_sequence` rõ ràng (vd: ["file_agent","page_agent"])
                - quyết định finalize ngay hoặc tiếp tục theo sequence để tăng grounding/citation
        → History tool shortcut (theo quyết định orchestrator):
            · count/area/report_pdf/report_excel từ chat_history khi context đủ dữ kiện
            · Nếu count=0 hoặc confidence thấp → fallback xuống page-level
        → Folder Agent (Gemini 2.5 Flash, được orchestrator gọi như 1 tool)
            · Input: list file.short_summary trong session scope
            · Chỉ answer nếu là câu hỏi overview rất chung chung
            · Nếu plan cho phép early-exit ở folder-level: "answer" → save history, trả thẳng
            · "go_to_file" → chọn ≤3 files
        → File Agent (Gemini 2.5 Flash, orchestrator tool) × N files
            · Input: query + files.summary (full)
            · Chỉ answer nếu summary có thông tin CỤ THỂ, còn lại escalate
            · Nếu plan cho phép early-exit ở file-level: "answer" → save history, trả thẳng
            · "go_to_page" → trả candidate_pages (page_number list) để load hẹp
        → Page Agent (orchestrator tool) — 2 stage:
            ┌─ Stage 1: Page Selector (Gemini 2.5 Flash)
            │   · Input: short_summary của page pool đã lọc (candidate_pages)
            │   · Output: top 5 page numbers liên quan nhất
            └─ Stage 2: Page Reasoner (Gemini 3.1 Pro)
                · Input: FULL context_md của ≤5 pages đã chọn (không cắt)
                         + chat history (last 5 turns)
                · Output: câu trả lời chi tiết + pages_used
                · need_tool detect: keyword match (優先) + Pro output
        → Tool (nếu cần):
            · count → count_tool (DXF exact / Gemini Pro Vision)
            · area  → area_tool  (catalog / Gemini Pro Vision / Gemini 2.5 Flash)
            · viz   → viz_tool   (vẽ boxes lên ảnh trang)
        → Image early-answer (nếu orchestrator cho phép):
            · preferred_tool=count → chạy count trực tiếp trên ảnh upload
            · preferred_tool=area  → chạy area trực tiếp trên ảnh upload
            · preferred_tool=none  → chạy general vision QA trực tiếp trên ảnh upload
            · nếu không đủ confidence/evidence thì orchestrator replan và tiếp tục xuống page-level
        → Save turn → MongoDB chat_history (giữ 5 turns gần nhất, $slice=-5)
      search:
        → chuyển sang Search pipeline (xem mục 9)
```

**Gemini calls per query (worst case):**
| Step | Model | Calls |
|------|-------|-------|
| Router | Flash | 1 |
| Folder Agent | Flash | 1 |
| File Agent | Flash | 1–3 |
| Page Selector | Flash | 1 |
| Page Reasoner | **Pro** | 1 |
| Tool count (optional) | Flash×1 (obj desc) + Pro×1 (vision) | 0–2 |
| **Total** | | **5–9** |

**Chat history (MongoDB collection `chat_history`):**
```
{
  _id: session_id_or_folder_id,
  turns: [                    // $slice=-5 → chỉ giữ 5 turns gần nhất
    { role_user: "...", role_assistant: "...", ts: ISODate },
    ...
  ]
}
```
- Ưu tiên 1 chat history cho mỗi `session_id`; fallback theo `folder_id` nếu không dùng session
- Chatbot nhớ ngữ cảnh 5 câu hỏi trước để trả lời follow-up
- Tool shortcut ưu tiên reuse dữ kiện từ history cho các câu hỏi follow-up để giảm số bước agent
- Page Selector và Page Reasoner đều nhận chat history để duy trì context

### 8. File/Folder management (delete_pipeline.py)
```
Delete file:
  1. Qdrant: xóa vectors filter file_id
  2. MongoDB: xóa pages + file doc
  3. Rebuild folder.summary từ remaining files (short_summary)

Delete folder:
  1. Qdrant: xóa vectors của tất cả files
  2. MongoDB: xóa pages + files + folder doc
  3. MongoDB: xóa chat_history của folder
```
→ Khi thêm file mới: upload_pipeline tự rebuild folder summary sau Step 9

### 9. Search pipeline (tính năng riêng, độc lập với chatbot — không dùng Gemini)
```
User query
  → Nếu có ảnh:
      1) title-block-first lookup:
         · trích drawing_no / drawing_title / project từ ảnh
         · match deterministic với files.title_block_index
         · nếu match đủ điểm → trả kết quả ngay (retrieval_mode=title_block_index)
      2) nếu không match → fallback vector search
  → Cohere embed (search_query, 1024-dim)
  → Qdrant top-K=100 (filter score ≥ 0.2, optional folder/file filter)
  → Fetch page + file metadata từ MongoDB
  → Return top-N=15: {file_name, page_number, image_url, vector_score}
```
Search trả raw results để frontend tự hiển thị, **không có agent tổng hợp câu trả lời**.

### 9b. Title-block lookup (ảnh thuộc file nào)
```
User upload ảnh + hỏi "ảnh này ở file nào?"
  → Trích xuất title_block metadata từ ảnh upload (drawing_no/title/project)
  → So khớp nhanh với files.title_block_index trong scope session/folder
  → Nếu match đủ điểm:
      trả file_name + page_number + citation ngay
  → Nếu không match:
      fallback sang orchestrator/page-level/similarity flow bình thường
```

Backfill cho dữ liệu cũ:
```bash
python -m cad_pipeline.scripts.backfill_title_block_index
python -m cad_pipeline.scripts.backfill_title_block_index --folder-id <folder_id>
python -m cad_pipeline.scripts.backfill_title_block_index --file-id <file_id>
```

---

## Serve ảnh local qua web (USE_S3=false)

Khi `USE_S3=false`, `image_url` trong MongoDB là local path (`data/images/...`). Để web app hiện ảnh khi trỏ domain, cần mount `StaticFiles` trong FastAPI:

```python
# api/app.py — thêm vào
from fastapi.staticfiles import StaticFiles
from cad_pipeline.config import LOCAL_IMAGES_DIR

app.mount("/images", StaticFiles(directory=LOCAL_IMAGES_DIR), name="images")
```

Sau đó khi lưu `image_url` vào MongoDB, build URL dạng HTTP thay vì local path:

```python
# trong upload_pipeline.py (khi USE_S3=false)
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
image_url = f"{BASE_URL}/images/{file_id}/pages/page_{page_number}.png"
```

Thêm vào `.env`:
```env
API_BASE_URL=https://your-domain.com   # domain thật khi deploy
```

**Kết quả:** ảnh trong MongoDB có URL dạng `https://your-domain.com/images/{file_id}/pages/page_1.png` → trỏ domain là hiện ngay, không cần cloud storage.

---

## Git push note (SSH key)

Repo này đang dùng GitHub SSH. Nếu môi trường tự nhận sai account (ví dụ báo `denied to hungdang97`), dùng explicit key để push:

```bash
# Push submodule trước
GIT_SSH_COMMAND='ssh -i "/mnt/data8tb/notex/.ssh/id_ed25519" -o IdentitiesOnly=yes' \
git -C "Chatbotsysteminterface" push origin main

# Push repo chính sau
GIT_SSH_COMMAND='ssh -i "/mnt/data8tb/notex/.ssh/id_ed25519" -o IdentitiesOnly=yes' \
git push origin main
```

Thứ tự push chuẩn: **submodule trước, parent repo sau** để commit pointer luôn hợp lệ.

---

## Reuse từ parent project

| Component | Path trong parent |
|-----------|------------------|
| Layout model | `layout_detect/models/checkpoints/cad_layout_v7_swapsplit/model_final.pth` (ResNet-101 FPN, 5 classes) |
| Symbol DB | `symbol_db/symbols_enriched.json` (2723 symbols, 21 groups) |
| Symbol groups | `symbol_db/symbol_groups.json` |
| Unit room catalog | `symbol_db/unit_room_catalog.json` |
| Object descriptions | `cad_pipeline/object_descriptions.json` (94 objects, 11 categories) |

## 21 symbol groups
apartment_unit, door, window, stair_ramp, elevator_escalator,
toilet_bathroom, kitchen_appliance, furniture, pipe_plumbing,
valve, structural, wall_partition_lgs, electrical_lighting,
fire_safety, hvac_ventilation, accessibility, annotation_dimension,
hatch_pattern, floor_plan_layout, landscape_outdoor, unknown_misc

---

## FastAPI endpoints (port 8001)
```
# Upload
POST   /upload                          Upload + index file (async)
GET    /upload/{job_id}/status          Poll upload progress
GET    /notifications                   List user notifications (persisted)
PATCH  /notifications/{id}/read         Mark one notification read/unread
PATCH  /notifications/read-all          Mark all notifications as read

# Chatbot (Q&A agent pipeline + chat history)
POST   /qa                              Chatbot query → Folder→File→Page→Tool
GET    /folders/{id}/chat-history       Lấy 5 turns gần nhất
DELETE /folders/{id}/chat-history       Xóa chat history của folder

# Search (tính năng riêng, trả raw results)
POST   /search                          Semantic search → top-N pages by vector score

# Data
GET    /folders                         List folders
GET    /folders/{id}/files              List files
GET    /files/{id}/pages                List pages

# Delete
DELETE /files/{file_id}?folder_id=...   Xóa file: MongoDB + Qdrant + rebuild folder summary
DELETE /folders/{folder_id}             Xóa folder: tất cả files + pages + vectors + chat history

# Tools (dùng trong chatbot hoặc gọi trực tiếp)
GET    /tools/count/groups              List symbol groups
GET    /tools/count?keyword=valve       Count symbols (symbol DB lookup)
GET    /tools/area/units                All unit types + areas
GET    /tools/area/units/{label}        Unit detail + room breakdown
POST   /tools/count/context             Count in page context (LLM)
POST   /tools/area/context              Area from page context (LLM)
```

---

## Recent logic updates (2026-04-22)

### 1) Upload notification center (persisted by user)
- Added backend notification persistence (`notifications` collection in MongoDB).
- Each upload now writes lifecycle notifications:
  - `processing` when upload starts (read=true by default),
  - `done` when indexing finishes (read=false),
  - `error` when upload fails (read=false).
- Added APIs:
  - `GET /notifications`,
  - `PATCH /notifications/{id}/read`,
  - `PATCH /notifications/read-all`.
- Frontend now has a bell icon notification center on Home:
  - unread badge,
  - dropdown list,
  - mark-one / mark-all read,
  - reload-safe because data is loaded from DB per user.

### 2) Long upload UX (3-minute detach)
- In upload modal, if processing exceeds 3 minutes:
  - modal auto-closes,
  - user sees "still processing in background",
  - polling continues in background,
  - completion/failure toast is shown when final status arrives.

### 3) Create session search parity
- `Create New Chat Session` search now:
  - runs name-based suggest first,
  - falls back to semantic search if suggest returns empty.
- Added duplicate session name guard in create modal:
  - if session name already exists (case-insensitive), creation is blocked and user is warned.

### 4) Image QA routing refinement
- Title-block lookup shortcut is now controlled by orchestrator plan (`allow_title_block_lookup`),
  not hardcoded keyword rules.
- This avoids misrouting image-content questions (e.g. "what is in this image")
  into file-match search, while still allowing source-file lookup intent.

### 5) Home upload simplification
- Removed folder upload CTA from Home upload section (file-only upload UI).
- Updated supported formats in UI:
  - `PDF, DOC, DOCX, XLS, XLSX, PNG, JPG, JPEG, WEBP, GIF, BMP, TIF, TIFF`.

### 6) Upload dependency fallback hardening
- Added robust image-load fallback in upload pipeline:
  - OpenCV path first (`cv2.imread`) when available,
  - Pillow fallback when OpenCV runtime is partial/stub.
- Added graceful layout-detection fallback:
  - if Detectron2 import fails, disable layout detection for remaining pages,
  - continue indexing flow without failing the job.

### 7) Notification polling optimization
- Frontend notifications polling is no longer a fixed-interval background loop.
- Client now continues polling only while there is at least one `processing` upload notification.
- This reduces repetitive `/notifications` traffic when there are no active uploads.

### 8) Upload/session behavior for existing folders
- Uploading a file into an existing folder now creates a new chat session per upload
  instead of reusing a folder-bound local session.
- Session sources persist with backend `chat_sessions.file_ids` using the uploaded `file_id`
  from `/upload/{job_id}/status`.

### 9) Upload fail-fast for LLM quota errors
- Upload pipeline now treats fatal LLM quota/billing errors (e.g. 429 `RESOURCE_EXHAUSTED`)
  as hard failures.
- If a page summary or block output contains quota-exhausted signals, upload raises an exception,
  job status becomes `error`, and notification status is updated to `Upload failed`.
- Frontend upload flow only navigates/creates session on `status=done`, so quota failures remain
  in error notification path.

### 10) Environment diagnostics + object description recovery
- Added `python -m cad_pipeline.scripts.check_env` (and `--strict`) to validate:
  - runtime modules (`cv2`, `numpy`, `PIL`, `fitz`, `pymongo`, `qdrant_client`)
  - optional modules (`detectron2`, `google.genai`, `cohere`)
  - key files and environment variables.
- Pinned OpenCV in `requirements.txt` to `opencv-python-headless==4.10.0.84`.
- Recreated and expanded `cad_pipeline/object_descriptions.json` baseline to 94 entries
  to restore object-description lookup flow.
