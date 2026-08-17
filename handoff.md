# Handoff

**jamslam** turns a noisy phone recording of an old-time fiddle jam into sheet
music containing only the fiddle melody.

This document is the orientation. It says what exists, what actually works, what
was tried and rejected with the measurement that rejected it, and where to pick
up. Detail lives in [`docs/RESEARCH.md`](docs/RESEARCH.md) (extraction) and
[`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) (the plan and the open problems).

---

## Status in one paragraph

The pipeline is complete and runs end to end: audio in, MusicXML/MIDI/ABC/PDF
out, with diagnostics at every stage and 122 passing tests. On synthetic audio
with known answers it performs respectably — pitch accuracy 0.81–0.88, key
1.00, tempo error 0.7%. **On the five real jam recordings the output is not yet
a usable transcription.** Two of the five produce readable, structured scores;
three produce an unsectioned stream of notes with every note flagged uncertain.
The single blocking problem, traced repeatedly and from several directions, is
that the extracted line is not reliably the fiddle.

---

## What is built

```
audio ─ preprocess ─ melody extraction ─ pitch cleaning ─ note segmentation
                                                              │
        beat tracking + tempo ────────────────────────────────┤
                                                              ▼
                                    quantization ─ form detection ─ consensus
                                                              │
                          key inference ─ score export ───────┘
                                                              ▼
                                     MusicXML · MIDI · ABC · PDF · diagnostics
```

Roughly 10,700 lines across 35 modules. Every stage takes its own config section
(`src/fiddle/config.py`) so any decision can be ablated from the command line
with `--config overrides.json`.

| area | module | notes |
|---|---|---|
| melody extraction | `melody/` | Basic Pitch (default), Essentia Melodia, pYIN, custom |
| cleaning | `pitch_clean.py` | octave correction, vibrato, jump gating — each switchable |
| segmentation | `segment.py` | pitch-change splitting, onset-based rearticulation |
| rhythm | `rhythm.py` | beat tracking, tempo-octave folding, quantization |
| form | `form.py` | section discovery, meter and barring, crooked bars |
| consensus | `consensus.py` | voting across repeated passes — the project's central bet |
| key | `key.py` | modal-aware (major/mixolydian/dorian/minor) |
| export | `score.py`, `abc_io.py` | MusicXML, MIDI, ABC, per-measure time signatures |
| banjo | `banjo.py` | Scruggs three-finger arranger with tablature |
| identification | `identify.py` | interval n-grams, IDF, per-tune null model |
| evaluation | `eval/` | separate metrics, never one opaque number |

Supporting tools: `tools/build_site.py` (status page), `tools/export_pdf.py`
(engraved PDFs via abcjs + headless Chromium), `tools/render_audio.py`
(fluidsynth playback, including original-vs-transcription A/B),
`tools/measure_posteriorgram.py` (the front-end diagnostic).

### Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,plots]"
.venv/bin/fiddle-transcribe recording.m4a --plots
```

Basic Pitch pins `numpy<2` and pulls TensorFlow, so it needs its own venv;
backend resolution falls back to Essentia loudly rather than silently.

---

## What works, measured

Synthetic corpus, exact ground truth, four difficulty tiers:

| metric | value |
|---|---|
| pitch accuracy | 0.81 (1.00 on clean takes) |
| pitch class | 0.85 |
| key + mode | **1.00** |
| tempo error | 0.7% |
| form accuracy | 0.75 |
| onset accuracy | 0.47 |
| duration accuracy | 0.49 |
| sequence similarity | 0.62 |
| octave errors | 0.033 |

Real recordings — `bebop_2` and `bebop_5` produce genuine scores: sections
marked with repeat barlines, real rhythmic variety, roughly half the notes
confident. `bebop_2` fits on one page.

---

## What does not work

Three of five real recordings (`bebop_1`, `bebop_4`, `memory_of_home`) find no
section structure, fall back to an unsectioned transcription, and flag every
note uncertain. The cause is not form detection. It is that their extracted
lines are too pitch-concentrated for sections to differ:

| recording | notes within ±1 semitone of median | IQR | form |
|---|---|---|---|
| bebop_2 | 24% | 12.0 | ✓ |
| bebop_5 | 34% | 5.1 | ✓ |
| bebop_4 | 31% | 7.0 | fails |
| bebop_1 | **39%** | **4.0** | fails |
| memory_of_home | 42% | 7.3 | fails |

`bebop_1` has half its line inside a four-semitone band; a fiddle tune spans an
octave or more.

**There is no ground truth for any real recording.** Every real-audio number in
this repository is a proxy, and proxies have misled this project repeatedly (see
below). Getting eight bars of one recording transcribed by ear would convert the
entire proxy stack into a measurement.

---

## The three findings worth carrying forward

**1. The melody is lost at thresholding, not at selection.** Basic Pitch emits
per-frame activations and then thresholds them into notes. Pitch-class entropy
by stage, against 2.40 bits / 53% for real old-time tunes:

| stage | entropy | top-2 pitch classes | energy kept |
|---|---|---|---|
| raw posteriorgram | 3.55 | 24.7% | 100% |
| thresholded at 0.5 | 1.58 | 86.7% | **2.9%** |
| after selection | 1.55 | 90% | — |

The default discards 97% of the energy and what survives is the loudest,
sustained content — guitar and banjo on the tonic and fifth. Selection adds only
four further points. **Any plan that reranks Basic Pitch's note events is
defeated before it starts**, because the notes are not in the pool. Matching
against the *un-thresholded* posteriorgram is not.

**2. A section length incommensurate with the tune's period manufactures
structure.** Its phase slides forward each block, so even lags return to phase
and odd ones do not, and clustering reads that as an A/B contrast — at several
times the quality the *correct* length reaches, where everything is in phase and
so everything looks alike. This was the entire content of every `AAB…` result
this project reported on real audio before it was caught. Incommensurate lengths
are no longer generated.

**3. Consensus must vote, never take a union.** Rearticulated repeated notes
were preserved by breaking a run of slots wherever *any* pass started a note.
With four passes of fifty notes over 128 slots, 96% of slots got marked, no run
ever merged, and a tune of 94 eighths and 17 quarters came out as 184
sixteenths. It was a bug that got worse the more evidence it was given.

---

## Tried and rejected, with the measurement

Keep these; each cost a real experiment.

| approach | verdict |
|---|---|
| **Chroma tune retrieval** | No discrimination: every tune scored 0.486–0.518 |
| **pYIN on real audio** | 0–2% voiced frames; its voicing detector never fires in a mix |
| **Frame-level Viterbi** | Locking fell to 8.7% but repetition collapsed sevenfold |
| **Sustain mask + Basic Pitch** | Monotonically worse; they are substitutes, not complements |
| **Voice-path selector** | Follows one voice as designed, and is worse: pitch 0.86 → 0.67 |
| **Melodic prior reranking** | Made concentration *worse* (90% → 91%) — notes aren't in the pool |
| **Time tolerance in contour similarity** | Cluster quality 0.089 → 0.091 |
| **Register-based section split** | Sound idea (the "low part"/"high part"), but the failing recordings have no register variation to read |
| **Interloper note filter** | Built for the duration bug, wasn't its cause; shipped off |
| **Source separation (Demucs)** | **Untested, not disproven** — weights unreachable from this environment |

---

## Two hazards that cost real time

**Proxies pointed the wrong way three times.** Lowering the Basic Pitch
threshold made repetition fall 0.29 → 0.19 while ground-truth note accuracy
*rose*. A bass-coupling control failed open and reported a clean pass on a
corpus that had no bass at all. `bebop_2`'s "strong" form result turned out to
be an artifact of its melody having collapsed onto one note. **Prefer the corpus
with known answers; treat every real-audio number as a hypothesis.**

**The corpus could not test what it was asked.** Every difficulty tier put the
fiddle at gain 0.9–1.0 against accompaniment of at most 0.46, so "keep the
loudest note" was close to correct *by construction* — and the corpus duly
scored it above every alternative while being structurally incapable of testing
one. A `buried` tier now exists (fiddle 0.55, guitar 0.85, banjo 0.80), which is
the condition the real recordings are in. **Measure melody-selection work there.**

---

## Where to pick up

The direction is **identify the tune against a catalog, then adapt it** — match
first, then let the recording override the known setting where it genuinely
differs (crooked bars, this player's variations, repeat counts). This is better
posed than free transcription and it sidesteps the two problems above: a matched
tune *supplies* the form that inference has now failed to find three times, and
identification only has to be right about *which tune*, not every note.

1. **Source the catalog.** This is the binding constraint and it is practical,
   not technical. thesession.org's database — 55,093 settings with real old-time
   overlap — carries a licence explicitly prohibiting processing with Large
   Language Models; it was downloaded, read, and deleted unused. Ryan's Mammoth
   Collection (1883) is properly public domain but its host is unreachable here.
   Only PyPI and `raw.githubusercontent.com` respond. **The highest-value single
   input is the name of the online library these five recordings came from** —
   it bounds the search to a few hundred candidates and probably contains the
   answers, which would hand us the ground truth this project has never had.
2. **Match against the posteriorgram**, not the note events, over transpositions
   and a tempo range.
3. **Validate on synthetic audio first**, where the answer is known, before
   believing any match on a real recording.
4. **Then align and adapt.**
5. **Try Demucs** in an environment with network access — still the largest
   untested lever.

---

## Repository map

```
src/fiddle/          pipeline (35 modules)
tests/               122 tests, written as statements about old-time music
tools/               site builder, PDF exporter, audio renderer, diagnostics
docs/RESEARCH.md     extraction findings, measured
docs/NEXT_STEPS.md   the plan, the open problems, why each fix failed
docs/index.html      generated status page (GitHub Pages: Settings → Pages → /docs)
reports/             raw output of every measurement cited anywhere
datasets/tunes/      memory_of_home is the gold-standard worked example
work/                scratch: real audio, per-run outputs (gitignored)
```

Every number in this document is reproducible from `reports/` or by re-running
the command in the corresponding commit message.
