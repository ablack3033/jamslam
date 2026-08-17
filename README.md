# jamslam

Infer the **canonical fiddle melody** from a recording of an old-time jam, and
produce clean sheet music containing only that melody.

**Status: Phase 1 (research CLI) is working and measured.** There is no API and
no frontend, deliberately — see [Findings](#findings) for the evidence that
should decide whether to build them.

---

## Quickstart

```bash
git clone <this repo> && cd jamslam
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,essentia,plots]"

# ffmpeg is needed for phone recordings (.m4a). Everything else is pip-installable.
sudo apt-get install -y ffmpeg     # or: brew install ffmpeg
```

Check which melody backends came up:

```bash
.venv/bin/fiddle-transcribe backends
```

### Transcribe a recording

```bash
.venv/bin/fiddle-transcribe recording.m4a --plots
```

Writes `recording.pitch.csv`, `recording.notes.json`, `recording.musicxml`,
`recording.mid`, `recording.abc` and `recording.diagnostics.png`, and prints:

```
key           D major  (confidence 0.97)
meter         4/4  (confidence 0.37)
tempo         123.0 BPM
form          AABB  (8 bars/section, confidence 0.22)
notes         151 canonical, 6 flagged uncertain
melody        essentia backend, 68% of frames voiced

least confident notes:
  beat   23.75  C#5  61%   alternatives: D5 50%
  beat   29.75  D5   64%   alternatives: C#5 25%, F#5 25%
  ...
```

Useful flags: `--melody-backend {essentia,pyin,basicpitch}`, `--key "A mixolydian"`,
`--meter 2/4`, `--no-consensus`, `--config overrides.json`.

### Measure it

```bash
# Synthetic corpus with exact ground truth (no audio files needed)
.venv/bin/fiddle-transcribe eval

# Your own recordings
.venv/bin/fiddle-transcribe eval --dataset datasets/tunes

# Compare design decisions against each other
.venv/bin/fiddle-transcribe ablate --json reports/ablation.json
```

### Tests

```bash
.venv/bin/python -m pytest -q          # 63 tests, ~35s
```

---

## Findings

Measured on the synthetic corpus (5 tunes × 3 difficulty levels, exact ground
truth). Full numbers in [`reports/`](reports/).

**What works.**

- **Predominant-melody extraction is not the bottleneck.** On clean audio the
  extracted and segmented pitch sequence matched ground truth *note for note*
  over the first bars, and the melody stays in the right register through the
  full band mix. Essentia's `PredominantPitchMelodia` is doing its job.
- **Beat tracking is excellent on this material.** Measured section boundaries
  landed 32.00, 32.02, 31.99 beats apart against a true 32 — old-time's steady
  guitar-and-bass pulse is a gift, and tempo error is under 1%.
- **Key inference works, including modal tunes.** Mixolydian is detected as
  mixolydian and given its parent major's key signature.
- **Octave errors are essentially absent** (rate < 0.02) despite the synthetic
  banjo deliberately doubling the melody an octave down.

**What does not work yet.**

- **Form detection is the bottleneck**, and it gates everything downstream. When
  the section length or offset is wrong, note-level metrics collapse even though
  the underlying melody was extracted correctly — you can see this directly in
  runs with 0.87 pitch accuracy but 0.15 onset accuracy, which is the signature
  of a correct melody shifted by a beat.
- **Confidence is not yet calibrated.** AUC hovers near 0.6 where it needs to be
  0.8+, and the uncertain-note flags currently have low precision. The product
  claim ("correct 2–5 flagged notes") depends entirely on this number, so it is
  the thing to fix after form.

**Verdict on the central hypothesis.** Not yet established. The evidence so far
supports the *first half* of it — constrained range and simple rhythms do make
extraction and beat tracking tractable — but the payoff from repetition and
consensus cannot be claimed until form detection is reliable, because consensus
only ever sees what form gives it. The apparatus to settle it now exists.

Three structural findings worth carrying forward, each of which cost a
measurement to learn:

1. **Similarity-based form analysis is completely flat in offset.** Shifting
   every block by the same amount leaves all pairwise similarities unchanged, so
   clustering can find the *period* of the repetition but never its *phase*.
   Phase needs external evidence; we currently use "the recording starts at the
   top of the tune", which is weaker than it should be.
2. **A half-section clusters exactly as cleanly as a whole one**, because half a
   section repeats too. No similarity-based objective can separate them. What
   does separate them is the domain prior that sections come in runs of exactly
   two: a half-length hypothesis yields runs of four, a double-length hypothesis
   runs of one. Encoding that fixed the single largest error class.
3. **Exact slot matching is too brittle for human performances.** Two passes of
   the same section routinely land a grid slot apart; requiring exact alignment
   made identical sections look ~30% similar and collapsed form detection on
   noisy audio. A ±1 slot tolerance roughly doubled the separation between
   same-section and different-section pairs.

---

## Where this deviates from the original spec

Stated plainly, because each of these was a deliberate decision.

**1. MusicXML is a poor ground-truth format; ABC is used instead.** The corpus
is only valuable at 20–50 recordings, and that only happens if writing ground
truth is cheap. An A part is one line of ABC a fiddler can type from memory; the
equivalent MusicXML is hundreds of lines nobody will proofread. Most of the
repertoire is already published in ABC. MusicXML ground truth still works
(`load_ground_truth` accepts both) — this is about the cost of the common case.
The parser validates that every section is a whole number of bars, so a typo
fails loudly rather than silently corrupting the corpus.

**2. "Detecting 2/4 vs 4/4" is not a well-posed problem.** The two are metrically
nested: the same performance notated in 2/4 with 16 bars and in 4/4 with 8 bars
is the same audio, and the tradition writes the same tune both ways. Any
confident answer is false precision. We infer a weak preference, cap the reported
confidence at 0.75, and make it overridable. What *is* well-posed and matters far
more is the **downbeat phase** — get that wrong and every barline is wrong, which
a musician notices instantly. Most of `meter.py` is about phase, not meter.

**3. Form detection must not consume the downbeat phase — it should produce it.**
The spec's pipeline order puts meter before form. We tried that and it failed:
onset-energy phase detection was wrong by one beat, which translated every note
and made it impossible for bar-aligned section boundaries to ever coincide with
real ones. Since quantization is a pure translation, phase can be applied *after*
form analysis at no cost, and a section boundary is by definition a downbeat —
far stronger evidence than onset energy. This reordering was the single largest
accuracy improvement made.

**4. Confidence should not be presented as a probability.** The spec's example
(`G 52%`, `D 98%`) implies calibrated likelihoods. Melodia's confidence is an
unbounded salience magnitude, and agreement across four passes is not a
probability. What we can honestly promise, and what the product actually needs,
is **correct ranking**: notes scored low must be wrong more often than notes
scored high. The eval harness measures exactly that (AUC, flag precision/recall)
and treats it as a headline metric rather than a footnote.

**5. Consensus needs a gate and an off switch.** Averaging passes that were
mis-segmented is worse than not averaging at all, so passes that fail to align
are excluded and recorded as outliers, and the whole stage degrades to
"use the single best pass". Without the off switch its benefit is unmeasurable,
which is why `--no-consensus` exists and is part of the ablation.

**6. A jam gives far more evidence than the spec's `A1 A2 B1 B2`.** Jams play a
tune four to eight times through, so there are typically 8–16 observations of the
A section, not two. The synthetic corpus reflects this (`AABBAABB` by default),
and it makes the consensus premise considerably more plausible than the spec
suggests — assuming form detection can deliver those observations.

**7. Basic Pitch cannot share the environment.** It pins `numpy<2` and pulls
TensorFlow 2.15. Rather than hold the whole project back, it sits behind the
`MelodyExtractor` interface with a lazy import and an isolated-venv install note.
Adapting a polyphonic note transcriber to this pipeline also requires a
melody-*selection* rule (we take the upper voice), which is an assumption stated
in the module rather than hidden — it is part of what any Melodia-vs-BasicPitch
comparison would actually be measuring.

---

## Architecture

Each stage is a module with a narrow interface, replaceable without touching
anything downstream.

```
Audio ──> AudioPreprocessor ──> MelodyExtractor ──> PitchContour
                                (essentia | pyin | basicpitch)
                                        │
                                        v
                                   PitchCleaner ──> NoteSegmenter ──> RawNotes
                                                          ^
Audio ──> BeatTracker/RhythmAnalyzer ──> beats, onsets ───┘
                    │
                    v
              quantize ──> TimedNotes ──> FormAnalyzer ──> sections + downbeat phase
                                                                │
                                                                v
                                            ConsensusBuilder ──> canonical Tune
                                                    │
                                     KeyAnalyzer ───┤
                                                    v
                                            ScoreExporter ──> MusicXML + MIDI + ABC
```

| module | responsibility |
|---|---|
| `domain.py` | typed musical model; confidence and provenance on every note |
| `config.py` | every tunable parameter, in one place, so ablations are one line |
| `audio.py` | decode anything to mono float32; normalize. No denoising. |
| `melody/` | `MelodyExtractor` protocol + Essentia / pYIN / Basic Pitch backends |
| `pitch_clean.py` | eight individually switchable cleaning stages |
| `segment.py` | contour → notes, using onsets to split rearticulated repeats |
| `rhythm.py` | beats, tempo-octave correction, quantization to simple durations |
| `meter.py` | meter preference and downbeat phase |
| `form.py` | section discovery via slot similarity + repeat-structure priors |
| `consensus.py` | per-slot weighted voting across aligned passes |
| `key.py` | modal-aware key inference from pitch-class distribution |
| `confidence.py` | the single definition of per-note confidence |
| `score.py` | music21 notation, MusicXML/MIDI/ABC export with repeat barlines |
| `pipeline.py` | wiring only — no DSP, no musical logic |
| `eval/` | metrics and corpus runners |
| `corpus/` | synthetic jam renderer with exact ground truth |

Design rules held throughout: MIDI is an export format and never the internal
model; continuous measurements are preserved alongside their discrete musical
interpretation, so re-keying or re-quantizing never requires re-analyzing audio;
every stage records what it did into `TranscriptionResult` for diagnosis.

### Domain model

`Note` carries `confidence`, `agreement`, `alternatives` (runners-up with
weights) and `provenance` (which passes voted for it). `TimedNote` keeps both
`start_beats`/`duration_beats` *and* the original `start_sec`/`duration_sec`,
because quantization is a hypothesis that may need revising. `TuneSection` keeps
the `SectionObservation`s it was distilled from, so a UI can eventually show
"3 of 4 passes played F#, one played F".

---

## Evaluation methodology

Metrics are reported **separately and never collapsed into one number** — a
transcription with correct pitches but a wrong beat phase is a rhythm bug, and
one with correct rhythm and octave-displaced pitches is an extraction bug. Those
need different fixes.

| metric | what it means |
|---|---|
| `pitch` | fraction of aligned notes with the exact right MIDI pitch |
| `pclass` | same, ignoring octave — the **gap between these two is the octave-error rate** |
| `onset` | fraction of truth notes matched within 0.3 beats |
| `dur` | fraction of matched notes with the right notated duration |
| `seq` | 1 − normalized edit distance over the melodic sequence |
| `key` / `meter` / `tempo%` | exact match, exact match, relative error |
| `form` | section pattern, compared modulo how many times the jam went round |
| confidence AUC | P(a wrong note scores below a correct one). 0.5 = useless |
| flag precision/recall | of notes flagged uncertain, how many were wrong; and vice versa |

Predicted and truth notes are matched by **global Needleman–Wunsch alignment**,
not nearest-neighbour: one inserted note early in a section would otherwise
misalign everything after it and report near-zero accuracy for a nearly correct
transcription. Sections are paired by *content*, not by label, since section
letters are arbitrary.

---

## Riskiest assumptions

Ordered by how much they would cost if false.

1. **That real jam audio behaves like the synthetic corpus.** Untested — this
   environment has no network access to fetch recordings. Real rooms have
   comb-filtering, a fiddle that wanders off mic, players who stop and restart,
   and conversation. This is the largest unknown by a wide margin.
2. **That form detection can be made reliable.** Everything downstream depends
   on it, and it is currently the weakest stage. If sections cannot be found
   robustly on real audio, the consensus premise — the project's actual
   differentiator — never gets a chance to pay off.
3. **That confidence can be calibrated well enough to be actionable.** The whole
   product proposition is "fix 2–5 flagged notes". At the current AUC (~0.6) the
   flags are barely better than chance, so a user would have to check everything,
   which is no better than transcribing by hand.
4. **That one melodic line dominates.** In a real circle two fiddlers play
   slightly different settings simultaneously. Melodia returns one contour; if
   the two settings genuinely diverge, the "canonical melody" may not be
   well-defined at all.
5. **That the tune starts at the top of the recording.** Currently used to pin
   the downbeat phase. Someone hitting record mid-tune produces a constant shift
   — a degradation rather than a breakdown, but a visible one.
6. **That quantization to a 16th grid is sufficient.** Old-time swings. Notated
   even eighths are frequently played closer to 60/40, which may systematically
   bias onsets.

---

## What to test it against

The corpus that would actually settle this, in priority order:

1. **5–10 recordings of the same tune** (say Soldier's Joy) from different jams,
   different rooms, different phones. This isolates recording conditions from
   tune difficulty, and it is the fastest way to find out whether real-world
   acoustics break the extractor.
2. **10–20 different tunes from one jam**, same room and same mic position.
   Isolates tune difficulty from recording conditions.
3. **Deliberate hard cases**: a crooked tune (Sandy River Belle, Yew Piney
   Mountain); a modal tune (Old Joe Clark, Cold Frosty Morning); a cross-tuned
   fiddle (Black Mountain Rag in AEAE); two fiddlers playing different settings
   at once; a tune in a key other than D/G/A.
4. **Bad recordings on purpose**: phone in a pocket, phone across the room,
   someone talking through the B part.

Take 2–3 minutes of audio, which gives 4–8 passes of each section. Write ground
truth as ABC using the fastest path: transcribe, fix the flagged notes in the
emitted `.abc`, save as `expected.abc`.

### Success and failure criteria

Judged on **real** recordings of type 1 and 2 above, at `jam`-like quality:

| | pitch accuracy | confidence AUC | form |
|---|---|---|---|
| **Success** — build the API and UI | > 0.90 | > 0.80 | correct on > 80% |
| **Promising** — keep improving the engine | 0.75–0.90 | 0.65–0.80 | > 60% |
| **Failure** — rethink the approach | < 0.75 | < 0.65 | < 50% |

The success bar is set by the product claim: a 32-note A part at 0.90 pitch
accuracy has ~3 wrong notes, which is the "correct 2–5 notes" experience, *but
only if the confidence flags actually point at them* — hence the joint AUC
requirement. Below 0.75, a user is checking every note, and transcribing by hand
is faster.

An explicit failure mode worth naming: if pitch accuracy is high but confidence
AUC stays near 0.5, the transcription is good but not *trustable*, and the right
response is to invest in calibration rather than in extraction.

---

## Roadmap

Phase 1 (this) is done and measured. Before Phase 2:

1. Fix form detection — better offset evidence, and section-boundary refinement
   after clustering. This is the top priority and gates everything.
2. Calibrate confidence against the corpus; move the blend weights in
   `ConfidenceConfig` with evidence rather than by taste.
3. Collect 20–50 real recordings and re-measure. **Do not build the API or the
   UI before this number exists.**

Phase 2 (FastAPI) and Phase 3 (minimal web UI) are intentionally not started.
The transcription engine runs entirely without FastAPI and always will —
`pipeline.transcribe()` is the whole interface an HTTP layer would need.
