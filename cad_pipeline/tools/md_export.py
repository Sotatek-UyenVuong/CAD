"""md_export.py — Markdown export helpers (PDF/DOCX via Pandoc)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def inject_front_matter(
    markdown: str,
    title: str = "Report",
    author: str = "CAD Pipeline",
    date: str | None = None,
    toc: bool = False,
) -> str:
    lines = [
        "---",
        f'title: "{title}"',
        f'author: "{author}"',
    ]
    if date:
        lines.append(f'date: "{date}"')
    if toc:
        lines += ["toc: true", "toc-depth: 2"]
    lines += ["---", ""]
    return "\n".join(lines) + (markdown or "")


def convert_to_pdf(
    md_path: str | Path,
    out_path: str | Path,
    title: str = "Report",
    author: str = "CAD Pipeline",
    date: str | None = None,
    toc: bool = False,
) -> bool:
    md_path = Path(md_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not md_path.exists():
        return False
    if shutil.which("pandoc") is None:
        return False

    src_text = md_path.read_text(encoding="utf-8")
    enriched = inject_front_matter(src_text, title=title, author=author, date=date, toc=toc)
    tmp_md = md_path.with_suffix(".frontmatter.md")
    tmp_md.write_text(enriched, encoding="utf-8")

    cmd = [
        "pandoc",
        str(tmp_md),
        "-o",
        str(out_path),
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=1in",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return proc.returncode == 0 and out_path.exists()
    finally:
        tmp_md.unlink(missing_ok=True)
