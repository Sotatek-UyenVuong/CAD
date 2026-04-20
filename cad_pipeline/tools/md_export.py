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


def convert_to_docx(
    md_path: str | Path,
    out_path: str | Path,
    title: str = "Report",
    author: str = "CAD Pipeline",
    date: str | None = None,
    toc: bool = False,
    reference_doc: str | Path | None = None,
) -> bool:
    md_path = Path(md_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not md_path.exists():
        return False
    if shutil.which("pandoc") is None:
        return _convert_to_docx_python(md_path, out_path, title=title)

    src_text = md_path.read_text(encoding="utf-8")
    enriched = inject_front_matter(src_text, title=title, author=author, date=date, toc=toc)
    tmp_md = md_path.with_suffix(".frontmatter.md")
    tmp_md.write_text(enriched, encoding="utf-8")

    cmd = [
        "pandoc",
        str(tmp_md),
        "-o",
        str(out_path),
    ]
    if reference_doc:
        ref_path = Path(reference_doc)
        if ref_path.exists():
            cmd.extend(["--reference-doc", str(ref_path)])
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return proc.returncode == 0 and out_path.exists()
    finally:
        tmp_md.unlink(missing_ok=True)


def _convert_to_docx_python(
    md_path: Path,
    out_path: Path,
    title: str,
) -> bool:
    """Fallback DOCX export using python-docx when pandoc is unavailable."""
    try:
        from docx import Document  # type: ignore
    except Exception:
        return False

    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return False

    doc = Document()
    doc.add_heading(title or "Report", level=1)

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            doc.add_paragraph("")
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
            continue
        if line.startswith("- "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
            continue
        if line.startswith("|") and line.endswith("|"):
            # Keep markdown table rows readable in fallback mode.
            doc.add_paragraph(line)
            continue
        doc.add_paragraph(line)

    try:
        doc.save(str(out_path))
        return out_path.exists()
    except Exception:
        return False
