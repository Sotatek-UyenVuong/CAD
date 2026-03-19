#!/usr/bin/env python3
"""md_export.py — Convert Markdown → DOCX or PDF (professional CAD report style).

Uses Pandoc with custom templates for clean, professional output.

Usage:
  python3 md_export.py report.md                  # → DOCX + PDF
  python3 md_export.py report.md --docx           # → DOCX only
  python3 md_export.py report.md --pdf            # → PDF only
  python3 md_export.py report.md -o out/          # → output directory
  python3 md_export.py report.md --title "CAD分析レポート" --toc

Requirements:
  pandoc, xelatex (texlive-xetex), Noto Sans CJK JP font
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Template paths (same directory as this script) ─────────────────────────
_HERE = Path(__file__).parent
DOCX_TEMPLATE = _HERE / "templates" / "cad_report_template.docx"
TEX_TEMPLATE  = _HERE / "templates" / "cad_report.tex"

# ── Pandoc defaults ────────────────────────────────────────────────────────
PANDOC_COMMON = [
    "--standalone",
    "--wrap=none",
]

PANDOC_DOCX = [
    "--to=docx",
    f"--reference-doc={DOCX_TEMPLATE}",
]

PANDOC_PDF = [
    "--to=pdf",
    "--pdf-engine=xelatex",
    f"--template={TEX_TEMPLATE}",
    "--variable=lang:ja",
    "--variable=fontsize:11pt",
]

# Stack size for large files (pandoc is Haskell — needs +RTS for big inputs)
# 256 MB stack — handles files up to ~50 MB
PANDOC_RTS = ["+RTS", "-K256m", "-RTS"]

# Files larger than this switch to weasyprint path for PDF
LARGE_FILE_BYTES = 2 * 1024 * 1024   # 2 MB
# DOCX truncation threshold (very large tables are barely usable in Word)
DOCX_TRUNCATE_BYTES = 1 * 1024 * 1024  # 1 MB

# Weasyprint CSS for large-file PDF
_WEASYPRINT_CSS = """\
@import url('file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');
* { font-family: "Noto Sans CJK JP", "Noto Sans", sans-serif; font-size: 9pt; }
body { margin: 1.5cm 1.8cm; }
h1 { font-size: 18pt; border-bottom: 2px solid #1a3a6b; color: #1a3a6b; margin-top: 1em; }
h2 { font-size: 14pt; border-bottom: 1px solid #888; color: #333; margin-top: 0.8em; }
h3 { font-size: 11pt; color: #555; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
th { background: #1a3a6b; color: white; padding: 4px 6px; font-size: 8pt; }
td { border: 1px solid #ccc; padding: 3px 5px; font-size: 8pt; word-break: break-all; }
tr:nth-child(even) td { background: #f5f7fb; }
blockquote { border-left: 3px solid #e8a000; padding-left: 8px; color: #555; }
code { background: #f4f4f4; padding: 1px 3px; font-size: 8pt; }
hr { border: 1px solid #ddd; }
@page { size: A4; margin: 1.5cm 1.8cm;
  @top-center { content: string(doctitle); font-size: 8pt; color: #888; }
  @bottom-right { content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; } }
h1 { string-set: doctitle content(); }
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def check_deps() -> list[str]:
    missing = []
    if not shutil.which("pandoc"):
        missing.append("pandoc")
    if not shutil.which("xelatex"):
        missing.append("xelatex (texlive-xetex)")
    return missing


def _has_weasyprint() -> bool:
    return shutil.which("weasyprint") is not None


def _convert_pdf_via_weasyprint(md_text: str, out_path: Path,
                                title: str | None, timeout: int) -> tuple[bool, str]:
    """MD → HTML (pandoc) → PDF (weasyprint). Used for large files."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        css_file  = Path(tmpdir) / "style.css"
        html_file = Path(tmpdir) / "doc.html"

        css_file.write_text(_WEASYPRINT_CSS, encoding="utf-8")

        # Step 1: MD → HTML (with +RTS for large inputs)
        html_args = ["pandoc", "+RTS", "-K512m", "-RTS",
                     "--standalone", "--wrap=none",
                     "--to=html5", "--metadata", f"title={title or 'Report'}",
                     "-o", str(html_file)]
        try:
            r = subprocess.run(html_args, input=md_text, text=True,
                               capture_output=True, timeout=timeout // 2)
            if r.returncode != 0:
                return False, f"pandoc HTML: {r.stderr.strip()[:200]}"
        except subprocess.TimeoutExpired:
            return False, f"pandoc HTML: timed out"

        # Step 2: HTML → PDF (weasyprint)
        try:
            r = subprocess.run(
                ["weasyprint", str(html_file), str(out_path),
                 "--stylesheet", str(css_file)],
                capture_output=True, text=True, timeout=timeout
            )
            if r.returncode != 0:
                return False, f"weasyprint: {r.stderr.strip()[:200]}"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "weasyprint: timed out"
        except Exception as e:
            return False, str(e)


def truncate_large_md(md_text: str, max_bytes: int, label: str) -> str:
    """Truncate markdown to max_bytes while keeping structure intact.
    Cuts at a table row boundary to avoid broken markdown."""
    encoded = md_text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return md_text

    # Find safe cut point (end of a line) within max_bytes
    cut_text = encoded[:max_bytes].decode("utf-8", errors="ignore")
    cut_pos = cut_text.rfind("\n")
    if cut_pos > 0:
        cut_text = cut_text[:cut_pos]

    total_lines = md_text.count("\n")
    kept_lines  = cut_text.count("\n")
    pct = kept_lines / max(total_lines, 1) * 100

    cut_text += (
        f"\n\n---\n> ⚠ **出力が大きすぎるため先頭 {kept_lines:,} 行のみ表示 "
        f"({pct:.0f}% / 全 {total_lines:,} 行)。"
        f"完全データは {label} を参照。**\n"
    )
    print(f"    ⚠  Truncated to {kept_lines:,}/{total_lines:,} lines ({pct:.0f}%) for {label}")
    return cut_text


def inject_front_matter(md_text: str, title: str | None, author: str | None,
                        date: str | None, toc: bool) -> str:
    """Prepend/merge YAML front matter into the markdown."""
    # Check if front matter already present
    if md_text.startswith("---"):
        return md_text  # user's own front matter — don't touch

    lines = ["---"]
    if title:
        lines.append(f"title: '{title}'")
    if author:
        lines.append(f"author: '{author}'")
    lines.append(f"date: '{date or time.strftime('%Y-%m-%d')}'")
    if toc:
        lines.append("toc: true")
    lines.append("---\n")
    return "\n".join(lines) + md_text


def run_pandoc(args: list[str], label: str, timeout: int = 120) -> tuple[bool, str]:
    """Run pandoc and return (success, error_message)."""
    try:
        result = subprocess.run(
            ["pandoc"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"{label}: timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


def convert_to_docx(md_path: Path, out_path: Path,
                    title: str | None, author: str | None,
                    date: str | None, toc: bool) -> bool:
    file_size = md_path.stat().st_size
    size_str  = f"{file_size/1024/1024:.1f} MB" if file_size > 1024*1024 else f"{file_size/1024:.0f} KB"
    print(f"  [DOCX] {md_path.name} ({size_str}) → {out_path.name} …", end=" ", flush=True)

    tmp = md_path.with_suffix(".~tmp.md")
    try:
        md_text = md_path.read_text(encoding="utf-8")

        is_large = file_size > LARGE_FILE_BYTES

        # Very large files: truncate DOCX too (>2MB tables are barely usable in Word)
        if is_large:
            md_text = truncate_large_md(md_text, DOCX_TRUNCATE_BYTES, md_path.name)

        effective_toc = toc and not is_large
        prepared = inject_front_matter(md_text, title, author, date, effective_toc)
        tmp.write_text(prepared, encoding="utf-8")

        rts     = PANDOC_RTS if is_large else []
        pandoc_timeout = 300 if is_large else 120
        args    = PANDOC_COMMON + rts + PANDOC_DOCX + [str(tmp), "-o", str(out_path)]
        ok, err = run_pandoc(args, "DOCX", timeout=pandoc_timeout)

        if ok:
            sz = out_path.stat().st_size / 1024
            print(f"✅  ({sz:.0f} KB)")
            return True
        else:
            print("❌")
            print(f"     Error: {err[:300]}")
            return False
    finally:
        if tmp.exists():
            tmp.unlink()


def convert_to_pdf(md_path: Path, out_path: Path,
                   title: str | None, author: str | None,
                   date: str | None, toc: bool) -> bool:
    file_size = md_path.stat().st_size
    size_str  = f"{file_size/1024/1024:.1f} MB" if file_size > 1024*1024 else f"{file_size/1024:.0f} KB"
    is_large  = file_size > LARGE_FILE_BYTES

    # Very large files: PDF conversion is impractical for pure data tables
    # (xelatex bufsize=200000 per line; weasyprint times out on large tables)
    # → produce DOCX only; direct user to Excel for full data
    if is_large:
        print(f"  [PDF ] {md_path.name} ({size_str}) → ⏭  skipped (file > {LARGE_FILE_BYTES//1024//1024} MB)")
        print(f"         💡 Use 'python3 tools/to_excel.py' for full data export.")
        return True  # not a failure — intentional skip

    print(f"  [PDF ] {md_path.name} ({size_str}) → {out_path.name} …", end=" ", flush=True)
    tmp = md_path.with_suffix(".~tmp.md")
    try:
        md_text = md_path.read_text(encoding="utf-8")
        effective_toc = toc and not is_large
        prepared = inject_front_matter(md_text, title, author, date, effective_toc)
        tmp.write_text(prepared, encoding="utf-8")

        rts      = PANDOC_RTS if is_large else []
        toc_args = ["--toc"] if effective_toc else []
        timeout  = 300 if is_large else 180
        args = PANDOC_COMMON + rts + PANDOC_PDF + toc_args + [str(tmp), "-o", str(out_path)]
        ok, err = run_pandoc(args, "PDF", timeout=timeout)

        if ok:
            sz = out_path.stat().st_size / 1024
            print(f"✅  ({sz:.0f} KB)")
            return True
        print("❌")
        print(f"     Error: {err[:300]}")
        return False
    finally:
        if tmp.exists():
            tmp.unlink()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Markdown to professional DOCX / PDF (CAD report style)"
    )
    parser.add_argument("input", nargs="+",
                        help="Input .md file(s) — supports glob e.g. '*.md'")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: same as input file)")
    parser.add_argument("--docx",  action="store_true", help="Generate DOCX only")
    parser.add_argument("--pdf",   action="store_true", help="Generate PDF only")
    parser.add_argument("--title",  default=None, help="Document title")
    parser.add_argument("--author", default=None, help="Author name")
    parser.add_argument("--date",   default=None, help="Date (default: today)")
    parser.add_argument("--toc",   action="store_true", help="Include table of contents")
    args = parser.parse_args()

    # Default: both DOCX + PDF
    do_docx = args.docx or (not args.docx and not args.pdf)
    do_pdf  = args.pdf  or (not args.docx and not args.pdf)

    # Dependency check
    missing = check_deps()
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Install: sudo apt install pandoc texlive-xetex texlive-lang-japanese")
        sys.exit(1)

    # Collect input files
    md_files: list[Path] = []
    for pattern in args.input:
        p = Path(pattern)
        if "*" in pattern or "?" in pattern:
            md_files.extend(sorted(p.parent.glob(p.name)))
        elif p.exists():
            md_files.append(p)
        else:
            print(f"⚠  File not found: {pattern}")

    if not md_files:
        print("No .md files found.")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0
    t0 = time.time()

    print(f"\nConverting {len(md_files)} file(s) …\n")

    for md in md_files:
        dest = out_dir or md.parent

        if do_docx:
            out = dest / (md.stem + ".docx")
            if convert_to_docx(md, out, args.title, args.author, args.date, args.toc):
                ok_count += 1
            else:
                fail_count += 1

        if do_pdf:
            out = dest / (md.stem + ".pdf")
            if convert_to_pdf(md, out, args.title, args.author, args.date, args.toc):
                ok_count += 1
            else:
                fail_count += 1

    elapsed = time.time() - t0
    print(f"\n── Done in {elapsed:.1f}s  ✅ {ok_count}  ❌ {fail_count} ──")


if __name__ == "__main__":
    main()
