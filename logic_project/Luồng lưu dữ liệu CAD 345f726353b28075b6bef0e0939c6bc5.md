# Luồng lưu dữ liệu CAD

Được tạo bởi: uyên vương
Thời gian tạo: 17 tháng 4, 2026 16:39
Chỉnh sửa gần nhất bởi: uyên vương
Lần cập nhật gần nhất: 17 tháng 4, 2026 16:49

# I. TỔNG KIẾN TRÚC LƯU TRỮ

```
S3 (storage vật lý)
    - folder/
        - file/
            - original file
            - page images
            - cropped images to block (image, diagram)

MongoDB (metadata + context)
    - folder / file / page / block

Qdrant (vector search)
    - embedding per page
```

---

# II. S3 STRUCTURE (rất quan trọng)

## 📁 Cấu trúc đề xuất:

```
s3://bucket/
    folders/
        {folder_id}/
            files/
                {file_id}/
                    original/
                        file.pdf

                    pages/
                        page_1.png
                        page_2.png

                    blocks/   (optional nhưng nên có)
                        page_1/
                            diagram_1.png
                            table_1.png
```

---

## 🔗 URL ví dụ:

```
https://s3/.../folders/folder_1/files/file_1/pages/page_5.png
```

---

## ✅ Quy tắc:

- **folder_id = logical grouping**
- **file_id = mỗi file upload**
- **page = unit nhỏ nhất cho Q&A**

---

# 🗄️ III. MONGODB SCHEMA

## 1. Collection: `folders`

```json
{
  "_id": "folder_id",
  "name": "Electrical Drawing A",
  "summary": "Folder chứa bản vẽ điện...",
  "created_at": "...",
  "updated_at": "..."
}
```

---

## 2. Collection: `files`

```json
{
  "_id": "file_id",
  "folder_id": "folder_id",
  "file_name": "drawing_A.pdf",
  "file_url": "s3://.../original/file.pdf",
  "summary": "Bản vẽ hệ thống điện tầng 1",
  "tags": ["electrical"],
  "total_pages": 10,
  "created_at": "..."
}
```

---

## 3. Collection: `pages` ⭐ (quan trọng nhất)

```json
{
  "_id": "page_id",
  "file_id": "file_id",
  "folder_id": "folder_id",

  "page_number": 5,

  "image_url": "s3://.../pages/page_5.png",

  "short_summary": "Trang này chứa sơ đồ điện...",

  "context_md": "...markdown full content...",

  "blocks": [
    {
      "type": "text",
      "content": "..."
    },
    {
      "type": "table",
      "content": "| ... |"
    },
    {
      "type": "diagram",
      "image_url": "s3://.../blocks/page_5/diagram_1.png",
      "description": "..."
    }
  ],

  "created_at": "..."
}
```

---

## 4. Collection: `blocks` (optional – nếu muốn scale sâu)

👉 Nếu bạn muốn query block-level sau này:

```json
{
  "_id": "block_id",
  "page_id": "page_id",
  "type": "diagram",
  "image_url": "...",
  "content": "...",
  "bbox": [x1, y1, x2, y2]
}
```

---

# 🔍 IV. QDRANT SCHEMA

## Collection: `page_vectors`

### Option đơn giản (1 vector):

```json
{
  "id": "page_id",
  "vector": [...],
  "payload": {
    "file_id": "...",
    "folder_id": "...",
    "page_number": 5
  }
}
```

---

## Option chuẩn hơn (multi-vector):

```json
{
  "id": "page_id",
  "vectors": {
    "text": [...],
    "image": [...]
  },
  "payload": {
    "file_id": "...",
    "folder_id": "...",
    "page_number": 5
  }
}
```

---

# 🔄 V. FLOW LƯU DATA (UPLOAD)

```
1. User upload file
2. Upload file → S3 (original)
3. Convert → images → upload S3 (pages)

4. Process từng page:
    - layout
    - OCR / Gemini
    - build context

5. Lưu Mongo:
    - page
    - file summary
    - folder summary

6. Tạo embedding:
    - text
    - image

7. Lưu Qdrant
```

---

# 🔍 VI. FLOW SEARCH

```
User query
   ↓
Embedding
   ↓
Qdrant search → page_id
   ↓
Mongo → lấy context_md + image_url
   ↓
Agent_Page xử lý
```

---

# 🤖 VII. FLOW Q&A

```
User hỏi
   ↓
Agent_Folder → files
   ↓
Agent_File → chọn file
   ↓
Agent_Page:
    - lấy page từ Mongo
    - reasoning
    - tool nếu cần
```

---

# ⚠️ VIII. QUY TẮC QUAN TRỌNG (đừng bỏ qua)

## 1. ID phải thống nhất

```
folder_id / file_id / page_id
→ dùng xuyên suốt Mongo + Qdrant + S3
```

---

## 2. Không lưu binary trong Mongo

```
❌ image base64
❌ file binary
✔ chỉ lưu URL S3
```

---

## 3. Context_md phải clean

```
- consistent format
- không random
- dễ parse
```

---

## 4. Qdrant chỉ lưu vector

```
❌ không lưu full context
✔ chỉ payload nhẹ
```

---

# 🎯 IX. TỔNG KẾT

👉 Hệ của bạn sẽ là:

```
S3:
    lưu file + image

Mongo:
    lưu logic + context

Qdrant:
    search nhanh
```

---

# 🚀 Nếu bạn muốn nâng cấp tiếp

Mình có thể:

- viết luôn **schema Mongo index (index gì để query nhanh)**
- hoặc design **multi-tenant (nhiều user, nhiều project)**
- hoặc tối ưu **chi phí lưu trữ + embedding (rất đáng làm)**