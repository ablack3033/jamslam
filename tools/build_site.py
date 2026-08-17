"""Generate the project status site served from docs/ via GitHub Pages.

The page's whole purpose is to show the *current measured* state of the
pipeline, so it is generated from real pipeline outputs rather than written by
hand -- otherwise it would drift from the code the first time anything changed.

Usage:

    fiddle-transcribe transcribe work/real/NAME.m4a --out work/final/ --plots
    python tools/build_site.py work/final

Reads the ``.abc``, ``.notes.json`` and ``.diagnostics.png`` a transcription run
produced, copies them under ``docs/results/`` and writes ``docs/index.html``.
"""

from __future__ import annotations

import collections
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS = DOCS / "results"

#: Recordings to show, in the order they appear on the page.
RECORDINGS = ("memory_of_home", "bebop_1", "bebop_2", "bebop_4", "bebop_5")

#: Audio we are allowed to publish. Only the gold-standard reference recording
#: is committed to this repository; the rest of the jam audio stays local, so
#: the page links a player for this one alone.
PUBLISHABLE_AUDIO = {"memory_of_home": "datasets/tunes/memory_of_home/recording.m4a"}

NOTE_TOKEN = re.compile(r"[_^=]?[A-Ga-g][,']*")


def abc_body(abc: str) -> str:
    return "\n".join(line for line in abc.splitlines() if line.startswith("|"))


def pitch_concentration(abc: str) -> tuple[int, list[tuple[str, float]]]:
    """How much of the extracted line sits on its few most common pitches.

    This is the single most diagnostic number on the page. A fiddle tune spreads
    across its scale; a line that is 90% two pitches is the tracker sitting on
    the tonic and fifth rather than following a melody.
    """
    tokens = NOTE_TOKEN.findall(abc_body(abc))
    if not tokens:
        return 0, []
    common = collections.Counter(tokens).most_common(3)
    return len(tokens), [(p, n / len(tokens)) for p, n in common]


def excerpt(abc: str, bars: int = 4) -> str:
    """The header plus the first few bars, for a legible rendered example.

    The no-form fallback emits the whole performance as one enormous line, which
    renders as an unreadable ribbon of notation, so every example is truncated
    and the full file offered as a download instead.
    """
    header = [line for line in abc.splitlines() if not line.startswith(("|", "P:"))]
    body = abc_body(abc)
    measures = [m for m in body.replace("|:", "|").split("|") if m.strip()]
    kept = "|".join(measures[:bars])
    return "\n".join(header) + f"\n|{kept}|"


#: Real old-time tunes put about 53% of their notes in two pitch classes,
#: measured over the catalog. The yardstick in both directions.
REAL_TUNE_TOP2 = 0.53


def verdict(summary: dict, concentration: list[tuple[str, float]]) -> tuple[str, str]:
    """A plain-language reading of one result, and a CSS class for it.

    Derived from the measurement rather than fixed prose, which went stale as
    soon as extraction improved -- it was still reporting a "drone collapse" on
    a line whose drone had gone.
    """
    top = sum(share for _, share in concentration[:2])
    unsectioned = not summary.get("form") or summary.get("form") == "A"
    tail = (" No section structure was found, so this is the raw performance "
            "rather than a canonical tune." if unsectioned else "")

    if top > 0.70:
        return (f"Melody collapsed onto the tonic and fifth -- {top:.0%} of "
                f"notes are just {concentration[0][0]} and "
                f"{concentration[1][0]}, against about {REAL_TUNE_TOP2:.0%} for "
                f"a real tune. Not a usable transcription." + tail, "bad")
    if top > 0.58:
        return (f"Still concentrated: {top:.0%} of notes in two pitches against "
                f"about {REAL_TUNE_TOP2:.0%} for a real tune. The drone lock is "
                f"broken but the line is not yet a melody." + tail, "warn")
    return (f"Pitch spread is now in the range of a real tune ({top:.0%} of "
            f"notes in two pitches, against about {REAL_TUNE_TOP2:.0%}), so the "
            f"line is no longer stuck on a drone. That does not mean the notes "
            f"are right -- nothing here is checked against ground truth."
            + tail, "warn")


def build(source: Path) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cards = []

    for name in RECORDINGS:
        abc_path = source / f"{name}.abc"
        json_path = source / f"{name}.notes.json"
        if not abc_path.exists() or not json_path.exists():
            print(f"skipping {name}: no output in {source}")
            continue

        abc = abc_path.read_text()
        summary = json.loads(json_path.read_text())["summary"]
        total, concentration = pitch_concentration(abc)
        message, level = verdict(summary, concentration)

        downloads = []
        for suffix in (".abc", ".musicxml", ".mid"):
            src = source / f"{name}{suffix}"
            if src.exists():
                shutil.copy(src, RESULTS / src.name)
                downloads.append(f'<a href="results/{src.name}">{suffix[1:]}</a>')

        diagnostics = source / f"{name}.diagnostics.png"
        has_diagnostics = diagnostics.exists()
        if has_diagnostics:
            shutil.copy(diagnostics, RESULTS / diagnostics.name)

        audio = ""
        if name in PUBLISHABLE_AUDIO:
            src = ROOT / PUBLISHABLE_AUDIO[name]
            if src.exists():
                shutil.copy(src, RESULTS / f"{name}.m4a")
                audio = (f'<audio controls preload="none" '
                         f'src="results/{name}.m4a"></audio>')

        cards.append(CARD.format(
            name=html.escape(name),
            title=html.escape(name.replace("_", " ").title()),
            level=level,
            message=html.escape(message),
            key=html.escape(str(summary.get("key", "?"))),
            meter=html.escape(str(summary.get("meter", "?"))),
            tempo=f"{summary.get('tempo_bpm', 0):.0f}",
            form=html.escape(str(summary.get("form") or "none")),
            bars=summary.get("bars_per_section", 0),
            n_notes=summary.get("n_notes", 0),
            n_uncertain=summary.get("n_uncertain", 0),
            voiced=f"{summary.get('voiced_fraction', 0):.0%}",
            concentration=", ".join(f"{p} {s:.0%}" for p, s in concentration),
            abc=html.escape(excerpt(abc)),
            downloads=" &middot; ".join(downloads),
            audio=audio,
            diagnostics=(f'<img loading="lazy" alt="diagnostic plots for {name}" '
                         f'src="results/{name}.diagnostics.png">'
                         if has_diagnostics else ""),
        ))

    DOCS.joinpath("index.html").write_text(PAGE.format(cards="\n".join(cards)))
    return DOCS / "index.html"


CARD = """
<article class="card">
  <header>
    <h3>{title}</h3>
    <p class="verdict {level}">{message}</p>
  </header>
  <dl class="facts">
    <div><dt>key</dt><dd>{key}</dd></div>
    <div><dt>meter</dt><dd>{meter}</dd></div>
    <div><dt>tempo</dt><dd>{tempo} BPM</dd></div>
    <div><dt>form</dt><dd>{form} ({bars} bars/section)</dd></div>
    <div><dt>notes</dt><dd>{n_notes}, {n_uncertain} uncertain</dd></div>
    <div><dt>voiced</dt><dd>{voiced} of frames</dd></div>
    <div><dt>top pitches</dt><dd>{concentration}</dd></div>
  </dl>
  {audio}
  <div class="score" data-abc="{abc}"></div>
  <p class="downloads">Full output: {downloads}</p>
  <details><summary>Diagnostic plots</summary>{diagnostics}</details>
</article>
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JamSlam &mdash; status</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>&#127925;</text></svg>">
<!-- abcjs is vendored rather than loaded from a CDN: the page then has no
     external dependencies, so it renders identically offline and cannot break
     when someone else's CDN does. MIT licensed, see vendor/abcjs-LICENSE.md. -->
<script src="vendor/abcjs-basic-min.js"></script>
<style>
  :root {{
    --bg: #fbfaf8; --fg: #1c1a17; --muted: #6b655c; --line: #e0dcd4;
    --card: #ffffff; --bad: #b3261e; --bad-bg: #fdecea;
    --warn: #8a5a00; --warn-bg: #fdf3e0; --ok: #1e6b3a; --ok-bg: #e9f5ed;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #16150f; --fg: #ece8e0; --muted: #9c968b; --line: #33302a;
      --card: #1e1c16; --bad: #f2b8b5; --bad-bg: #3a1512;
      --warn: #f0c070; --warn-bg: #3a2c10; --ok: #9fd8b3; --ok-bg: #12301d;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 62rem; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }}
  h1 {{ font-size: clamp(1.9rem, 4vw, 2.6rem); margin: 0 0 .3rem; letter-spacing: -.02em; }}
  h2 {{ font-size: 1.35rem; margin: 3rem 0 1rem; letter-spacing: -.01em; }}
  h3 {{ font-size: 1.1rem; margin: 0 0 .4rem; }}
  .lede {{ color: var(--muted); font-size: 1.05rem; max-width: 44rem; }}
  .banner {{
    margin: 1.75rem 0; padding: 1rem 1.15rem; border-radius: .6rem;
    background: var(--bad-bg); color: var(--bad);
    border: 1px solid color-mix(in srgb, var(--bad) 25%, transparent);
  }}
  .banner strong {{ display: block; font-size: 1.05rem; }}
  .banner p {{ margin: .4rem 0 0; color: inherit; opacity: .9; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
  .scroll {{ overflow-x: auto; }}
  th, td {{ text-align: left; padding: .55rem .7rem; border-bottom: 1px solid var(--line); }}
  th {{ font-weight: 600; color: var(--muted); font-size: .82rem;
        text-transform: uppercase; letter-spacing: .04em; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }}
  pre {{ background: var(--card); border: 1px solid var(--line); border-radius: .5rem;
         padding: .9rem 1rem; overflow-x: auto; font-size: .85rem; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: .7rem;
    padding: 1.25rem 1.35rem; margin: 1.1rem 0;
  }}
  .verdict {{ margin: 0; padding: .5rem .7rem; border-radius: .4rem; font-size: .92rem; }}
  .verdict.bad {{ background: var(--bad-bg); color: var(--bad); }}
  .verdict.warn {{ background: var(--warn-bg); color: var(--warn); }}
  .verdict.ok {{ background: var(--ok-bg); color: var(--ok); }}
  .facts {{ display: grid; gap: .1rem .9rem; margin: 1rem 0;
            grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); }}
  .facts div {{ display: flex; gap: .4rem; border-bottom: 1px solid var(--line);
                padding: .3rem 0; font-size: .88rem; }}
  .facts dt {{ color: var(--muted); min-width: 5.5rem; }}
  .facts dd {{ margin: 0; font-weight: 500; }}
  .score {{ overflow-x: auto; margin: .5rem 0; }}
  .score svg {{ max-width: 100%; }}
  audio {{ width: 100%; margin: .5rem 0; }}
  .downloads {{ font-size: .88rem; color: var(--muted); }}
  a {{ color: inherit; }}
  details {{ margin-top: .6rem; font-size: .9rem; color: var(--muted); }}
  details img {{ max-width: 100%; margin-top: .7rem; border: 1px solid var(--line);
                 border-radius: .4rem; }}
  footer {{ margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--line);
            color: var(--muted); font-size: .88rem; }}
</style>
</head>
<body>
<div class="wrap">

<h1>JamSlam</h1>
<p class="lede">Turning noisy iPhone recordings of old-time fiddle jams into sheet
music containing only the fiddle melody. This page reports what the pipeline
actually produces today, generated from real runs.</p>

<div class="banner">
  <strong>Status: not working end to end.</strong>
  <p>The pipeline runs, exports valid MusicXML, MIDI and ABC, and passes its
  tests. The <em>musical content is wrong</em>: on most recordings the extracted
  line collapses onto the tonic and fifth instead of following the fiddle. The
  scores below are shown as evidence of that, not as results.</p>
</div>

<h2>What it does</h2>
<pre>audio -&gt; melody extraction (Basic Pitch, register floor D4, loudest voice)
      -&gt; pitch cleaning -&gt; note segmentation
      -&gt; beat tracking -&gt; quantization
      -&gt; form detection (which stretches repeat) -&gt; consensus across repeats
      -&gt; MusicXML + MIDI + ABC</pre>
<p>The bet is <strong>consensus</strong>: a jam plays each part many times, those
passes are independent noisy measurements of one melody, so voting across them
should beat any single pass. That bet is not yet testable, because the melody
going into it is not the fiddle.</p>

<h2>The five real recordings</h2>
{cards}

<h2>Where it does work: synthetic audio</h2>
<p>On a synthetic corpus with exact ground truth &mdash; same code, same backend
&mdash; the machinery performs, which localises the failure to melody extraction
from real room recordings rather than to the pipeline downstream of it.</p>
<div class="scroll">
<table>
  <tr><th>metric</th><th>clean</th><th>meaning</th></tr>
  <tr><td>pitch accuracy</td><td>0.88 mean, 1.00 on clean takes</td>
      <td>correct notes</td></tr>
  <tr><td>form accuracy</td><td>0.80</td>
      <td>correct section structure (was 0.67)</td></tr>
  <tr><td>key</td><td>0.73</td><td>correct key and mode</td></tr>
  <tr><td>tempo error</td><td>0.7%</td><td>beat tracking</td></tr>
</table>
</div>

<h2>Two findings worth keeping</h2>
<p><strong>Form detection was reporting an artifact.</strong> A section length
that is not commensurate with the tune's period slides its phase forward each
block, so blocks an even number apart come back into phase and odd ones do not.
Clustering reads that as a clean A/B contrast. Measured on one recording: at
block lengths that are multiples of the 16-beat period, similarity is 0.50 flat
at every lag; at lengths that are not, it alternates 0.20/0.52 and wins the
search outright. Those lengths are no longer generated.</p>
<p><strong>What that exposed is the real blocker.</strong> At the
<em>correct</em> section length, the A part and the B part resemble each other
exactly as much as two passes of the A part do. The extracted line carries the
tune's periodicity but not enough detail to separate its sections &mdash; a
melody-extraction problem that no form prior can fix.</p>

<h2>Where the melody is actually lost</h2>
<p>Basic Pitch emits per-frame activations and only then thresholds them into
discrete notes. Measuring pitch-class entropy at each stage localises the
failure precisely:</p>
<div class="scroll">
<table>
  <tr><th>stage</th><th>pc-entropy</th><th>top-2 pitch classes</th>
      <th>energy kept</th></tr>
  <tr><td>raw posteriorgram</td><td>3.55 bits</td><td>24.7%</td><td>100%</td></tr>
  <tr><td>thresholded at 0.3</td><td>1.83</td><td>83.4%</td><td>9.8%</td></tr>
  <tr><td>thresholded at 0.5 (near default)</td><td>1.58</td><td>86.7%</td>
      <td><strong>2.9%</strong></td></tr>
  <tr><td>after melody selection</td><td>1.55</td><td>90%</td><td>&mdash;</td></tr>
  <tr><td><em>real old-time tunes, for scale</em></td><td><em>2.40</em></td>
      <td><em>53%</em></td><td>&mdash;</td></tr>
</table>
</div>
<p><strong>Thresholding discards 97% of the energy</strong>, and what survives is
the loudest, most sustained content &mdash; guitar and banjo on the tonic and the
fifth. Melody selection adds only four further points of concentration, so it is
not the culprit. The fiddle is in the recording; it sits below the confidence
threshold.</p>

<h2>What would move this next</h2>
<ol>
  <li><strong>Identify the tune against a catalog, then adapt it.</strong> The
  fiddle is present but weak &mdash; only 12.2% of voiced frames peak on
  anything other than the two dominant pitch classes &mdash; so no frame-by-frame
  rule will find it. Matching a known melody against the un-thresholded
  posteriorgram integrates that weak evidence across a whole tune. Once the tune
  is known, what remains is small: find where this performance departs from the
  known setting.</li>
  <li><strong>Ground truth.</strong> Eight bars of one recording transcribed by
  ear would convert every open question here into a measurement. Every number on
  this page is a proxy, and proxies are gameable. Successful identification
  would produce this automatically.</li>
  <li><strong>Source separation.</strong> Removing the bass and guitar stems
  before extraction attacks the problem at its root. Untested &mdash; the
  pretrained weights were unreachable from the build environment, so this is
  unproven rather than disproven.</li>
</ol>

<footer>
Generated from pipeline output by <code>tools/build_site.py</code>.
Source: <a href="https://github.com/ablack3033/jamslam">github.com/ablack3033/jamslam</a>.
Notation rendered with abcjs. Only the gold-standard reference recording is
published here; the other jam audio stays out of the repository.
</footer>

</div>
<script>
  document.querySelectorAll(".score").forEach(function (el, i) {{
    ABCJS.renderAbc(el, el.dataset.abc, {{
      responsive: "resize", staffwidth: 720, scale: 1.0
    }});
  }});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "final"
    print("wrote", build(source))
