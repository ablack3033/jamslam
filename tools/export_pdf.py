"""Engrave transcriptions to PDF.

music21 can emit PDF only by shelling out to MuseScore or LilyPond, neither of
which is available here, so notation is rendered with abcjs in headless Chromium
and printed to PDF. That also keeps the PDFs identical to what the status site
shows, since both go through the same renderer.

Usage:

    python tools/export_pdf.py work/final            # every .abc in the directory
    python tools/export_pdf.py work/final bebop_1    # just one

Writes alongside the source unless --out is given.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "docs" / "vendor" / "abcjs-basic-min.js"
CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

NOTE_TOKEN = re.compile(r"[_^=]?[A-Ga-g][,']*")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<script>{abcjs}</script>
<style>
  @page {{ size: A4 portrait; margin: 16mm 14mm 18mm; }}
  body {{ margin: 0; font: 11pt/1.5 Georgia, "Times New Roman", serif; color: #111; }}
  header {{ margin-bottom: 6mm; }}
  h1 {{ font-size: 17pt; margin: 0 0 1mm; }}
  .meta {{ font-size: 9.5pt; color: #444; margin: 0; }}
  .caveat {{
    font-size: 8.5pt; color: #7a2018; background: #fdecea;
    border: 0.4pt solid #e0b4ae; border-radius: 2mm;
    padding: 2.5mm 3mm; margin: 3mm 0 0;
  }}
  #score svg {{ max-width: 100%; display: block; break-inside: avoid; }}
</style></head>
<body>
<header>
  <h1>{title}</h1>
  <p class="meta">{meta}</p>
  <p class="caveat">{caveat}</p>
</header>
<div id="score"></div>
<script>
  // The exporter emits each section as one long ABC line, so without explicit
  // wrapping abcjs draws a single staff hundreds of bars wide and the print
  // clips it to whatever fits one page. wrap re-flows by measure instead.
  // Two options matter for print, and both were found by looking at the output:
  //   wrap           -- the exporter emits each section as one long ABC line, so
  //                     without it abcjs draws a single staff hundreds of bars
  //                     wide and printing clips it to whatever fits one page.
  //   oneSvgPerLine  -- one tall SVG will not break directly after the header
  //                     block, which left page one blank. Separate SVGs per
  //                     staff line let the pages fill naturally.
  ABCJS.renderAbc("score", {abc}, {{
    staffwidth: 680, scale: 0.92, paddingtop: 0,
    oneSvgPerLine: true,
    wrap: {{ minSpacing: 1.6, maxSpacing: 2.6, preferredMeasuresPerLine: 4 }},
  }});
</script>
</body></html>
"""


def describe(abc: str, summary: dict | None) -> tuple[str, str, str]:
    """Title, metadata line, and an honest caveat for one transcription."""
    title = next((l[2:].strip() for l in abc.splitlines() if l.startswith("T:")),
                 "Untitled")
    if not summary:
        return title, "", ""

    tokens = NOTE_TOKEN.findall(
        "\n".join(l for l in abc.splitlines() if l.startswith("|")))
    top = 0.0
    if tokens:
        counts = sorted({t: tokens.count(t) for t in set(tokens)}.values(),
                        reverse=True)
        top = sum(counts[:2]) / len(tokens)

    form = summary.get("form") or "none found"
    meta = (f"{summary.get('key', '?')} &middot; {summary.get('meter', '?')} &middot; "
            f"{summary.get('tempo_bpm', 0):.0f} BPM &middot; form {form} &middot; "
            f"{summary.get('n_notes', 0)} notes, "
            f"{summary.get('n_uncertain', 0)} flagged uncertain")

    if top > 0.6:
        caveat = (f"Not a usable transcription. {top:.0%} of the notes below are "
                  f"just two pitches — the tonic and the fifth — which "
                  f"means melody extraction locked onto a drone rather than "
                  f"following the fiddle. Printed as evidence of the failure.")
    else:
        caveat = ("Unverified against ground truth. No section structure was "
                  "found, so this is the raw performance rather than a canonical "
                  "tune, and every note is flagged uncertain.")
    return title, meta, caveat


def export(abc_path: Path, out_dir: Path, page) -> Path:
    abc = abc_path.read_text()
    name = abc_path.stem
    summary_path = abc_path.with_suffix(".notes.json")
    summary = (json.loads(summary_path.read_text()).get("summary")
               if summary_path.exists() else None)
    title, meta, caveat = describe(abc, summary)

    page.set_content(PAGE.format(
        abcjs=VENDOR.read_text(),
        title=html.escape(title),
        meta=meta,
        caveat=html.escape(caveat),
        # The title is already in the page header; leaving T: in would engrave
        # it a second time at the top of the first staff.
        abc=json.dumps("\n".join(l for l in abc.splitlines()
                                 if not l.startswith("T:"))),
    ), wait_until="load")
    page.wait_for_selector("#score svg", timeout=30_000)
    page.wait_for_timeout(400)

    out = out_dir / f"{name}.pdf"
    page.pdf(path=str(out), format="A4", print_background=True,
             margin={"top": "16mm", "bottom": "18mm",
                     "left": "14mm", "right": "14mm"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="directory holding .abc output")
    ap.add_argument("names", nargs="*", help="restrict to these stems")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.source.glob("*.abc"))
    if args.names:
        paths = [p for p in paths if p.stem in set(args.names)]

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page()
        for abc_path in paths:
            print("wrote", export(abc_path, out_dir, page))
        browser.close()


if __name__ == "__main__":
    main()
