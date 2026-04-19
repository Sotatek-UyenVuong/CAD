# CAD Pipeline — Standalone Sub-project

Full end-to-end pipeline for indexing and querying Japanese architectural CAD drawings (PDF/images).

---

## Architecture

```
UPLOAD FLOW
-----------
File (PDF/image)
  │
  ▼
pdf_to_images.py       → render pages @ 300 DPI
  │
  ▼
layout_detect.py       → Detectron2 model (text / table / diagram / title_block)
  │
  ▼
page_processor.py      → OCR (text/table) + Gemini Pro (diagram/title_block)
  │
  ▼
context_builder.py     → build Markdown context_md per page
  │
  ▼
embeddings.py          → OpenAI text-embedding-3-small
  │
  ├─► MongoDB           (page metadata + context_md)
  └─► Qdrant            (page vector index)

Q&A FLOW
--------
User Query
  │
  ▼
router.py              → classify: "qa" or "search"
  │
  ├─ qa ──► folder_agent → file_agent → page_agent → tools (count/area)
  └─ search ──► Qdrant search → Gemini re-rank → results

TOOLS
-----
count_tool.py          → count symbols via LLM or symbol_db
area_tool.py           → calculate floor areas via LLM or unit_room_catalog.json
```

---

## Setup

```bash
cd cad_pipeline
cp .env.example .env
# Fill in GEMINI_API_KEY, OPENAI_API_KEY, MARKER_API_KEY, DATABASE_URL, QDRANT_URL

pip install -r requirements.txt
```

> **Marker API**: OCR (text/table blocks) được xử lý qua Datalab.to cloud API.
> Lấy API key tại https://www.datalab.to → set `MARKER_API_KEY` trong `.env`.

---

## Run the API server

```bash
cd /path/to/CAD
python -m cad_pipeline.api.app
# or
uvicorn cad_pipeline.api.app:app --host 0.0.0.0 --port 8001 --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/upload` | Upload + index a file (async) |
| `GET`  | `/upload/{job_id}/status` | Check upload progress |
| `POST` | `/qa` | Ask a question |
| `POST` | `/search` | Semantic image/diagram search |
| `GET`  | `/folders` | List all folders |
| `GET`  | `/folders/{id}/files` | List files in folder |
| `GET`  | `/files/{id}/pages` | List pages in file |
| `GET`  | `/tools/count/groups` | List symbol groups |
| `GET`  | `/tools/count?keyword=valve` | Count symbols by keyword |
| `GET`  | `/tools/area/units` | All unit types + areas |
| `GET`  | `/tools/area/units/{label}` | Unit detail with room breakdown |

---

## Quick usage examples

### Upload a PDF
```bash
curl -X POST http://localhost:8001/upload \
  -F "file=@drawing.pdf" \
  -F "folder_id=folder_001" \
  -F "folder_name=Electrical Drawings"
```

### Ask a question
```bash
curl -X POST http://localhost:8001/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "バルブは何個ありますか？", "folder_id": "folder_001"}'
```

### Search for similar diagrams
```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pump diagram with flow arrows", "top_k": 5}'
```

### Count symbols
```bash
curl "http://localhost:8001/tools/count?keyword=valve&use_symbol_db=true"
```

### Get floor areas
```bash
curl "http://localhost:8001/tools/area/units/apt_unit_100A"
```

---

## Python usage

```python
# Upload
from cad_pipeline.pipeline.upload_pipeline import run_upload_pipeline
result = run_upload_pipeline(
    file_path="drawing.pdf",
    folder_id="folder_001",
    folder_name="Electrical Drawings",
)
print(result)  # {"file_id": ..., "total_pages": 10, "page_ids": [...]}

# Q&A
from cad_pipeline.pipeline.qa_pipeline import run_qa
answer = run_qa(query="総面積はいくらですか？", folder_id="folder_001")
print(answer["answer"])

# Search
from cad_pipeline.pipeline.search_pipeline import run_search
results = run_search(query="emergency exit diagram", top_k=5)

# Count
from cad_pipeline.tools.count_tool import run_count_tool
count = run_count_tool(query="valve", use_symbol_db=True)
print(count)  # {"count": 42, "unique_symbols": 5, ...}

# Area
from cad_pipeline.tools.area_tool import get_unit_area
info = get_unit_area("apt_unit_100A")
print(info["total_m2"])  # e.g. 162.97 m²
```

---

## Project structure

```
cad_pipeline/
├── config.py                  # all configuration from .env
├── .env.example
├── requirements.txt
├── README.md
├── core/
│   ├── pdf_to_images.py       # PDF → PNG pages
│   ├── layout_detect.py       # Detectron2 layout model wrapper
│   ├── page_processor.py      # OCR + Gemini per block
│   ├── context_builder.py     # build Markdown context
│   └── embeddings.py          # OpenAI embeddings
├── storage/
│   ├── mongo.py               # MongoDB CRUD
│   └── qdrant_store.py        # Qdrant vector CRUD
├── agents/
│   ├── router.py              # qa vs search classifier
│   ├── folder_agent.py        # Level 1 agent
│   ├── file_agent.py          # Level 2 agent
│   └── page_agent.py          # Core reasoning agent
├── tools/
│   ├── count_tool.py          # symbol counting (DB + LLM)
│   └── area_tool.py           # area calculation (catalog + LLM)
├── pipeline/
│   ├── upload_pipeline.py     # full upload flow
│   ├── qa_pipeline.py         # full Q&A flow
│   └── search_pipeline.py     # global semantic search
└── api/
    └── app.py                 # FastAPI server
```

---

## Reused from parent project

| Component | Source |
|-----------|--------|
| Layout detection model | `layout_detect/models/checkpoints/cad_layout_v7_swapsplit/model_final.pth` |
| Symbol database | `symbol_db/symbols_enriched.json` (2723 symbols, 21 groups) |
| Symbol groups | `symbol_db/symbol_groups.json` |
| Unit room catalog | `symbol_db/unit_room_catalog.json` |

## OCR Strategy

| Block type | Tool | Notes |
|------------|------|-------|
| `text` | Gemini 2.5 Flash vision | Transcribe text exactly as-is |
| `table` | Marker API (fast mode) | Cloud OCR → Markdown table |
| `diagram` | Gemini Pro vision | Description in original language |
| `title_block` | Marker API → Gemini Flash | OCR text → structured JSON |
| Page summary | Gemini Flash vision | 1-2 sentence overview |
