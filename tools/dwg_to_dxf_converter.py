#!/usr/bin/env python3
"""
DWG → DXF Converter (headless, server-friendly)

Correct ODA File Converter CLI format (7 args):
  ODAFileConverter <InputDir> <OutputDir> <OutputVersion> <OutputType>
                   <Recurse> <Audit> [filter]
  Example: ODAFileConverter /in /out ACAD2018 DXF 0 1 "*.DWG"

Usage:
  Single file : python3 dwg_to_dxf_converter.py drawing.dwg
  Batch folder: python3 dwg_to_dxf_converter.py /dwg_folder/ -o /output/
  Check tools : python3 dwg_to_dxf_converter.py --check
"""

import sys
import os
import subprocess
import shutil
import tempfile
import argparse
from pathlib import Path


# ── Tool detection ─────────────────────────────────────────────────────────────

def _find_binary(*names: str, extra_dirs: list[str] | None = None) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for base in ["/opt", "/usr/share", "/usr/local"] + (extra_dirs or []):
        for candidate in Path(base).rglob(names[0]):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def find_dwg2dxf() -> str | None:
    return _find_binary("dwg2dxf")


def find_oda_file_converter() -> str | None:
    # Prefer system install
    found = _find_binary("ODAFileConverter", extra_dirs=["/opt/ODA", "/usr/share/ODA"])
    if found:
        return found
    # Fall back to AppImage in project folder or common locations
    for d in [Path(__file__).parent, Path.home() / "Downloads", Path("/tmp")]:
        for f in sorted(d.glob("ODAFileConverter*.AppImage"), reverse=True):
            if os.access(f, os.X_OK):
                return str(f)
    return None


def find_xvfb_run() -> str | None:
    return shutil.which("xvfb-run")


def find_xdotool() -> str | None:
    return shutil.which("xdotool")


def check_tools() -> dict[str, str | None]:
    tools = {
        "dwg2dxf (LibreDWG)":        find_dwg2dxf(),
        "ODAFileConverter (CLI)":     find_oda_file_converter(),
        "xvfb-run (virtual display)": find_xvfb_run(),
        "xdotool (auto-click)":       find_xdotool(),
    }
    print("\n── Tool availability ─────────────────────────────────")
    for name, path in tools.items():
        status = f"✅  {path}" if path else "❌  not found"
        print(f"  {name:<35} {status}")
    print()
    return tools


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 120, env: dict | None = None) -> tuple[int, str]:
    try:
        e = {**os.environ, **(env or {})}
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)
        return result.returncode, (result.stderr or result.stdout).strip()
    except subprocess.TimeoutExpired:
        return 1, "timed out"
    except FileNotFoundError as exc:
        return 1, str(exc)


def _start_xvfb() -> tuple[subprocess.Popen | None, str]:
    """Start Xvfb on a free display, return (proc, display_str)."""
    xvfb = shutil.which("Xvfb")
    if not xvfb:
        return None, ""
    num = 201
    while Path(f"/tmp/.X{num}-lock").exists():
        num += 1
    disp = f":{num}"
    proc = subprocess.Popen(
        [xvfb, disp, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    import time; time.sleep(1.5)
    return proc, disp


def _auto_click_ok(display: str) -> None:
    """Background thread: wait for ODA info dialog and click OK."""
    import time, threading

    def _click():
        xdotool = find_xdotool()
        if not xdotool:
            return
        for _ in range(20):
            time.sleep(1)
            r = subprocess.run(
                [xdotool, "search", "--name", "ODA File Converter"],
                capture_output=True, text=True,
                env={**os.environ, "DISPLAY": display}
            )
            wid = r.stdout.strip().split()[-1] if r.stdout.strip() else ""
            if wid:
                env = {**os.environ, "DISPLAY": display}
                subprocess.run([xdotool, "windowfocus", "--sync", wid], env=env,
                               capture_output=True)
                time.sleep(0.3)
                subprocess.run([xdotool, "mousemove", "--window", wid, "240", "280"],
                               env=env, capture_output=True)
                time.sleep(0.2)
                subprocess.run([xdotool, "click", "1"], env=env, capture_output=True)
                return

    t = threading.Thread(target=_click, daemon=True)
    t.start()


# ── Converters ─────────────────────────────────────────────────────────────────

def convert_with_libredwg(dwg: Path, out_dir: Path) -> Path | None:
    binary = find_dwg2dxf()
    if not binary:
        return None
    dxf = out_dir / (dwg.stem + ".dxf")
    print(f"    [LibreDWG] dwg2dxf …")
    code, msg = _run([binary, "-o", str(dxf), str(dwg)])
    if code == 0 and dxf.exists():
        return dxf
    print(f"    ⚠  dwg2dxf: {msg}")
    return None


def convert_with_oda(dwg: Path, out_dir: Path) -> Path | None:
    """
    ODA File Converter — correct 7-arg CLI format:
      ODAFileConverter <InputDir> <OutputDir> <OutputVersion> <OutputType>
                       <Recurse> <Audit> [filter]
    """
    binary = find_oda_file_converter()
    if not binary:
        return None

    xvfb_proc, display = _start_xvfb()
    if not display:
        # Try current DISPLAY if available
        display = os.environ.get("DISPLAY", "")
    if not display:
        print("    ⚠  No display available. Install xvfb: sudo apt install xvfb")
        if xvfb_proc:
            xvfb_proc.terminate()
        return None

    print(f"    [ODA] ODAFileConverter (DISPLAY={display}) …")

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_in_path = Path(tmp_in)
        # Copy file (avoid symlink issues with non-ASCII filenames)
        dest = tmp_in_path / (dwg.stem + ".DWG")
        shutil.copy2(dwg, dest)

        # Start auto-click in background thread
        _auto_click_ok(display)

        # Correct format: InputDir OutputDir OutputVersion OutputType Recurse Audit [filter]
        cmd = [binary, str(tmp_in_path), str(out_dir),
               "ACAD2018", "DXF", "0", "1", "*.DWG"]
        code, msg = _run(cmd, timeout=60, env={"DISPLAY": display})

    if xvfb_proc:
        xvfb_proc.terminate()

    for ext in (".dxf", ".DXF"):
        dxf = out_dir / (dwg.stem + ext)
        if dxf.exists():
            return dxf

    if msg:
        print(f"    ⚠  ODA: {msg}")
    return None


# ── Core convert ───────────────────────────────────────────────────────────────

CONVERTERS = [convert_with_libredwg, convert_with_oda]


def convert(dwg: Path, out_dir: Path) -> Path | None:
    if not dwg.exists():
        print(f"  ❌  Not found: {dwg}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Converting: {dwg.name}")
    for fn in CONVERTERS:
        result = fn(dwg, out_dir)
        if result:
            print(f"    ✅  → {result}")
            return result
    _print_install_help()
    return None


def batch_convert(input_dir: Path, out_dir: Path) -> tuple[int, int]:
    dwg_files = sorted(input_dir.rglob("*.dwg")) + sorted(input_dir.rglob("*.DWG"))
    if not dwg_files:
        print(f"  No .dwg files found in {input_dir}")
        return 0, 0
    print(f"\n  Found {len(dwg_files)} DWG file(s)")
    ok, fail = 0, 0
    for dwg in dwg_files:
        rel = dwg.relative_to(input_dir)
        target_dir = out_dir / rel.parent
        if convert(dwg, target_dir):
            ok += 1
        else:
            fail += 1
    return ok, fail


def _print_install_help() -> None:
    print("""
  ── No converter found ────────────────────────────────────────────────────
  A) LibreDWG (pure CLI, no display needed):
       sudo add-apt-repository universe
       sudo apt install libredwg-tools

  B) ODA File Converter + virtual display:
       sudo apt install xvfb xdotool
       # ODA already installed at /usr/bin/ODAFileConverter
  ─────────────────────────────────────────────────────────────────────────
""")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DWG → DXF converter")
    parser.add_argument("input", nargs="?", help="DWG file or folder")
    parser.add_argument("-o", "--output", help="Output folder")
    parser.add_argument("--check", action="store_true", help="Show available tools")
    args = parser.parse_args()

    if args.check:
        check_tools()
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else None

    if input_path.is_dir():
        target = out_dir or input_path
        ok, fail = batch_convert(input_path, target)
        print(f"\n  Batch done — ✅ {ok} converted, ❌ {fail} failed")
        sys.exit(0 if fail == 0 else 1)
    elif input_path.suffix.lower() == ".dwg":
        target = out_dir or input_path.parent
        dxf = convert(input_path, target)
        if dxf:
            print(f"\n  Next:  python3 dxf_parser.py \"{dxf}\"\n")
            sys.exit(0)
        sys.exit(1)
    else:
        print(f"  ❌  Expected .dwg file or directory: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
