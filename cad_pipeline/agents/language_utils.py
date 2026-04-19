"""language_utils.py — lightweight query language detection for response control."""

from __future__ import annotations

import re


def detect_query_language(query: str) -> str:
    """Return one of: 'vi', 'ja', 'en'."""
    q = query or ""
    # Japanese scripts
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", q):
        return "ja"
    # Vietnamese diacritics and common words
    if re.search(r"[ăâêôơưđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]", q):
        return "vi"
    ql = q.lower()
    if any(w in ql for w in ["bao nhiêu", "là gì", "cho tôi", "giúp tôi", "trang", "bản vẽ", "tài liệu"]):
        return "vi"
    return "en"


def language_label(code: str) -> str:
    if code == "vi":
        return "Vietnamese"
    if code == "ja":
        return "Japanese"
    return "English"
