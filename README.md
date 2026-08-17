# jamslam

Infer the **canonical fiddle melody** from a recording of an old-time jam, and
produce clean sheet music containing only that melody.

**Status: Phase 1 (research CLI) is built and measured. It works on synthetic
audio, does not yet work on real jam recordings, and the synthetic corpus has
been shown to be an unreliable guide to real-audio behaviour.** There is no API and no
frontend, deliberately — see [Findings on real recordings](#findings-on-real-recordings),
which is the evidence that should decide what to do next.

A **status page** showing what the pipeline currently produces on all five real
recordings — rendered notation, diagnostics, and a plain verdict per recording —
is generated from real runs by `tools/build_site.py` into `docs/`. Enable GitHub
Pages with source *Deploy from a branch → `/docs`* to serve it.

---

## Quickstart

```bash
git clone <this repo> && cd jamslam
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,essentia,plots]"

# ffmpeg is needed for phone recordings (.m4a). Everything else is pip-installable.
sudo apt-get install -y ffmpeg     # or: brew install ffmpeg
```

### The default backend needs its own environment

**Basic Pitch is the default melody backend**, because it measured far better on
real jam recordings than anything else tried (see
[Best known approach](#best-known-approach-basic-pitch-with-a-register-floor)).
It pins `numpy<2` and pulls TensorFlow, so it *cannot* be installed alongside the
rest of this package. Give it its own environment:

```bash
python3 -m venv .venv-bp
.venv-bp/bin/pip install basic-pitch essentia -e .
.venv-bp/bin/fiddle-transcribe recording.m4a
```

The environment above works without it: an unavailable backend **falls back to
Essentia and says so on stderr**.

```
melody backend 'basicpitch' unavailable (basic-pitch is not installed in this
environment); falling back to 'essentia'
```

The fallback is deliberately loud. Choosing a backend is meaningful precisely
because it changes the result, so you should never be left believing you ran
Basic Pitch when you did not.

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

Useful flags: `--melody-backend {essentia,fiddle,pyin,basicpitch}`, `--key "A mixolydian"`,
`--meter 2/4`, `--no-consensus`, `--banjo`, `--identify`, `--config overrides.json`.

### Identify the tune

Matching a transcription against a catalog of known settings. This is primarily
a **validation** mechanism -- see [Validation by identification](#validation-by-identification).

```bash
# Against the small builtin catalog of common jam tunes
.venv/bin/fiddle-transcribe identify recording.m4a

# Against a real tune library (any directory of .abc files, or one tunebook)
.venv/bin/fiddle-transcribe identify recording.m4a --catalog /path/to/library

# Reuse an existing transcription instead of re-analyzing the audio
.venv/bin/fiddle-transcribe identify --from-json recording.notes.json
```

### Arrange a three-finger banjo part

Takes audio *or* an ABC melody, so it is useful now, before extraction is
reliable -- point it at a hand-corrected `.abc` and it produces a real part.

```bash
.venv/bin/fiddle-transcribe banjo tune.abc
.venv/bin/fiddle-transcribe banjo recording.m4a --capo 2 --roll forward_reverse
```

```
Banjo arrangement -- D major, 4/4, open-G tuning, capo 0

D|7 ----4 -7 -5 -4 -2 -4 -|0 ----0 -2 -4 -0 ----0 -|
B|---3 -------------------|---2 -------------2 ----|
G|------------------------|------------------------|
D|------------------------|------------------------|
g|------------------------|------------------------|
  M  i  M  M  M  M  M  M   M  I  M  M  M  M  I  M
```

Uppercase finger = melody note, lowercase = roll filler; T/I/M = thumb/index/middle.

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
.venv/bin/python -m pytest -q          # 88 tests, ~60s
```

---

## Findings on real recordings

**Five real phone recordings of jams (3.5–6.4 minutes each) were tested. The
pipeline does not currently produce a usable transcription of any of them.**
This is the headline result and it supersedes the synthetic numbers below.

| recording | key (inferred) | tempo | form | notes | uncertain | identification |
|---|---|---|---|---|---|---|
| bebop_1 | D major | 112 | **failed** | 606 | 59% | z = 1.5 |
| bebop_2 | E dorian | 123 | **failed** | 390 | 17% | z = −0.4 |
| bebop_4 | A mixolydian | 123 | **failed** | 483 | 66% | z = 2.0 |
| bebop_5 | A major | 123 | **failed** | 863 | 55% | z = 2.1 |
| memory_of_home | A major | 103 | ABBCD / 20 bars (bogus) | 457 | 14% | z = 2.7 |

Form detection failed outright on four of five, and returned its own search
bound on the fifth. No single recording clears the identification confidence bar
(z ≥ 6), against a matcher that scores a correct melody at z = 9–39 and still at
z = 12 with one note in seven corrupted.

### Do the transcriptions carry any real signal? Currently undecidable.

Three tunes were later named as present among these recordings — Old Joe Clark,
Cluck Old Hen and Sandy River Belle — and one of the three ranks first or second
out of twenty for every recording. A permutation test put that at p = 0.0004,
and **that number was reported here and is now retracted.**

It assumed a named tune was the *correct* answer for every recording. It is not:
`memory_of_home` is probably not Old Joe Clark, so the top-ranked match on that
recording is a false positive rather than a hit, and the test's premise fails.

Measuring the chance rates directly, on 200–300 random diatonic walks:

| | rank #1 on noise | rank ≤2 on noise |
|---|---|---|
| uniform expectation | 5.0% | 10.0% |
| Old Joe Clark | 5.3% | 15.0% |
| Sandy River Belle | 3.7% | 11.0% |
| Cluck Old Hen | — | 6.0% |
| best of the three | — | **31.0%** |

So the observed pattern breaks down into one plausible hit and one likely miss,
at comparable individual significance:

- `bebop_1` → Sandy River Belle at rank 1, chance rate **3.7%**. Tentatively
  confirmed, and suggestive on its own — but a single p ≈ 0.04 observation is
  not evidence.
- `memory_of_home` → Old Joe Clark at rank 1, chance rate **5.3%**. Believed
  wrong, which is exactly what a chance hit looks like.

One likely-true and one likely-false at the same significance is **no net
evidence either way**. The honest position is that this is undecidable on
current data, and settling it needs verified ground truth for even one
recording, not more statistics.

Worth noting what did survive: the permutation machinery itself was sound — its
assumed null (28.4%) matched the measured one (31.0%). The statistic was right;
the assumption fed into it was wrong. That is the more dangerous failure mode,
and it is why the chance rates above are now measured rather than assumed.

Diagnosing this produced three real bug fixes:

**The melody extractor was tracking the accompaniment.** `min_frequency` was
130 Hz (~C3), set below the fiddle deliberately so octave errors would stay
visible to the cleaner. On real jam audio that was a serious mistake: guitar,
bass and banjo own the 100–250 Hz band and are usually closer to a phone's
microphone than the fiddle is. The diagnostic plot for the gold-standard
recording shows a sparse contour parked on sustained low pitches, with one
section that is almost entirely a single low note. Raising the floor to 250 Hz
moves the median extracted pitch from MIDI 57 (an accompaniment drone) to MIDI
69 (the fiddle's register). That is now the default.

**Identification was matching on nothing, twice over.** First, without rarity
weighting the matcher ranked the same tune top for all five recordings at an
*identical* score, matching purely on the all-zeros interval pattern that
repeated notes produce. Fixing that was not enough: raw match scores are not
comparable *between* tunes, because a repetitive, stepwise tune matches
meaningless input far more readily than an arpeggiated one. Measured directly,
one catalog tune won **60 of 60** random-walk trials at a mean score of 0.394 --
higher than any real recording scored. Scores are now normalised against that
per-tune null baseline and reported as z.

**The MusicXML exporter crashed on the no-form fallback.** That path runs
whenever form detection fails, which on real audio is the common case -- so the
most-exercised real-world path had no test, because the synthetic corpus rarely
takes it.

The honest conclusion: on real audio the bottleneck is **upstream of everything
clever**. Consensus and form analysis cannot help if the contour does not
contain the melody. Fix extraction on real recordings first.

## Best known approach: Basic Pitch with a register floor

Full write-up in [`docs/RESEARCH.md`](docs/RESEARCH.md). Measured on `bebop_1`,
5m36s of real jam audio. `locked` is the share of frames holding one fixed
interval to the bass (lower is better); `REP` is excess self-similarity at the
tune's period (higher is better).

| approach | locked ↓ | REP ↑ |
|---|---|---|
| Melodia (default) | 63.5% | 0.287 |
| Melodia + sustain mask | 53.0% | 0.248 |
| frame-level Viterbi | 8.7% | 0.041 |
| contour-level DP | 24.4% | 0.173 |
| Basic Pitch + sustain mask | 28.3% | 0.284 |
| **Basic Pitch, floor D4, loudest voice** | **26.5%** | **0.286** |

Basic Pitch is the only approach that improves bass independence **without**
sacrificing repetition. Everything else traded one against the other, which is
why none of them counted as an improvement.

The reason is structural. Melodia computes a salience function and then commits
to a line internally, on generic criteria. Basic Pitch emits **discrete note
events for every voice** and leaves the selection to us — and selecting among
notes is far better posed than selecting among salience peaks, because notes
carry register, duration, amplitude and overlap. The rule that measured best is
deliberately blunt: **discard everything below D4, then take the loudest note
sounding at each instant.** Taking the *highest* note instead — the obvious
rule — measured worse on both, because it chases upper partials.

Stacking the sustain mask underneath it does **not** help — locking rises and
coverage falls monotonically, because the mask deletes signal Basic Pitch was
using. The two are substitutes, not complements: the mask helps Melodia, which
has no other way to tell the fiddle from the guitar, and is redundant under a
model that already separates the voices.

It cannot share the main environment (it pins `numpy<2` and pulls TensorFlow),
so it runs from a separate venv. Its model ships inside the wheel, so inference
needs no network — which also makes it the only candidate here viable for the
fully client-side deployment discussed above.

```bash
python -m venv .venv-basicpitch
.venv-basicpitch/bin/pip install basic-pitch -e .
.venv-basicpitch/bin/fiddle-transcribe recording.m4a --melody-backend basicpitch
```

## Melody selection is the real problem on real audio

Melodia's pitch *tracking* is good. Its melody *selection* is what fails on a
phone recording of a jam: it builds a salience function over the whole mix and
picks the strongest coherent contour, and in a real circle the guitar and bass
are often physically closer to the microphone than the fiddle.

Looking at every salience peak rather than Melodia's single choice shows three
layers in a representative window of `bebop_1`: a static line near A3, a static
line at D4, and a third layer around F#4-D5 that **moves melodically**. Melodia
selects the static D4. The fiddle is the moving line above it.

`--melody-backend fiddle` reuses Essentia's salience computation and contour
tracking -- the parts that work -- and replaces only the final selection, using
three things a generic selector does not know:

* **A melody moves; a drone does not.** A contour holding one pitch for a second
  or more is an open string or a held chord tone, however salient.
* **The melody is usually the top voice**, even when it is quieter.
* **A contour at a locked interval above a lower one is that lower one's
  harmonic**, not an independent line.

On `bebop_1` this moves time-spent-on-a-single-pitch from 42% to 27%, and
interval-locking to the bass from 63% to 41%. Better, not solved.

### A check that fails closed

`presence.py` asks, before transcribing, whether there is a melody here at all,
and the pipeline reports `melody_found`. Emitting a tidy score for a recording
whose melody was never found is the worst thing this system can do, because a
musician cannot distinguish it from success without checking every note -- the
work the product exists to avoid.

Two corrections are recorded in that module, both found by checking a result
that looked good:

* The measure originally asked whether the melody sat an octave or a fifth above
  the bass. That is confounded -- a fiddle playing in its low register over a
  root-position accompaniment genuinely does. It now asks whether the interval is
  **locked**, since a real melody's interval to the bass varies as it moves.
* It originally returned 0.0 when there was no trackable bass, which reads as
  "no problem found". The synthetic corpus has no trackable bass at all, so a
  control that never ran was reported as a clean pass. It now returns
  "not assessed" and fails closed. **The threshold remains unvalidated on real
  audio**, and is stated as a heuristic rather than a calibrated test.

### End-to-end on the five real recordings

| recording | presence check | form |
|---|---|---|
| bebop_1 | Melodia ✗ → BP ✓ | *no structure* (clusters separate by 0.012) |
| bebop_2 | Melodia ✗ → BP ✓ | 8 bars of 4/4, one bar of 3 (quality 0.44) |
| bebop_4 | Melodia ✗ → BP ✓ | *no structure* (0.071) |
| bebop_5 | Melodia ✗ → BP ✓ | *no structure* |
| memory_of_home | Melodia ✗ → BP ✓ | 6 bars of 4/4 (quality 0.081) |

The melody-presence check flips from 0 of 5 to **5 of 5** — it has never before
returned a clean verdict on real audio.

Form used to return a section length and an `AAB…` label sequence for all five.
Those were not weak versions of the right answer; they were an artifact of
searching section lengths incommensurate with the tune's period, which
manufactures alternating structure out of nothing. See
[`docs/RESEARCH.md`](docs/RESEARCH.md#form-the-phase-rotation-artifact) for the
measurement. Removing it leaves the real problem exposed: at the *correct*
section length, the A part and the B part resemble each other exactly as much as
two passes of the A part do. The extracted melody carries the tune's periodicity
but not enough detail to separate its sections.

So three of five now report no structure and fall back to an unsectioned
transcription. That is the honest outcome — a wrong split makes consensus
average unrelated music together, which is worse than not averaging at all.
Uncertain-note counts stay high, and no recording is confidently identified.

## Validation by identification

Hand-writing ground truth is the thing that makes a regression corpus expensive,
which is why the corpus is hard to grow. Tune identification attacks that
directly: if a transcription matches a known setting strongly, the match is
itself evidence the transcription is roughly right, and the catalog setting can
serve as approximate ground truth.

```
transcribe -> identify -> use the matched setting as expected.abc -> measure
```

How the matching works, and why:

- **Interval n-grams, not note sequences.** Comparing intervals is
  transposition-invariant by construction, which matters because a jam plays a
  tune in whatever key it likes and our own key inference may be off. Using
  n-grams rather than whole-sequence alignment makes it robust to insertions and
  deletions: a spurious note destroys only the few n-grams spanning it.
- **Rarity weighting.** Without it, melodically empty patterns dominate.
- **Null-model normalisation.** Each tune's score is measured against random
  diatonic walks and reported as z. Without this the ranking is dominated by
  whichever catalog tune is easiest to match by accident.
- **Matched against the flat note stream, not detected sections.** Form
  detection is the least reliable stage and fails on most real recordings;
  identification that depended on it would fail exactly when most needed.

Calibration of the matcher itself, from the test suite:

Scores are reported as **z**: standard deviations above what that particular
tune scores against meaningless stepwise input. Raw scores are not comparable
between tunes, so z is what ranks and what the confidence test uses.

| input | z |
|---|---|
| a catalog tune against itself | **9 – 39** |
| the same tune transposed | unchanged |
| one note in seven corrupted | **12.3**, still confident |
| random diatonic walks (30 trials) | none confident, no tune dominates |
| a single repeated note | not confident |
| **the five real recordings** | **−0.4 to 2.7 — none confident** |

So the matcher is not the weak link. Two sigma on a four-minute recording is not
ambiguity about *which* tune it is; it is evidence that the transcription
contains no more tune-specific information than noise.

**Caveat, stated plainly:** this environment has no network access, so the
builtin catalog is 20 common jam tunes entered by hand, and it may simply not
contain the tunes in these recordings. Point `--catalog` at a real library to
rule that out — it accepts any directory of `.abc` files.

## Crooked tunes

Old-time is full of tunes whose sections carry an odd bar — a 2-beat or 6-beat
bar inside an otherwise 4/4 section — so the section total is not a whole number
of bars. This is now handled end to end:

- **Search.** Section length is searched at one-beat resolution rather than
  one-bar. Previously a crooked section was not merely scored badly, it was
  never generated as a hypothesis at all, so the search silently returned the
  nearest wrong answer.
- **Barring.** The barring is chosen *from* the section length rather than given
  to it. 2/4 and 4/4 are metrically nested, so the same recording is correctly
  barred either way and beat stress can barely choose between them — but bar
  count can, because the 8-bar section is close to universal here. A 16-beat
  section is 8 bars only in 2/4; a 32-beat section only in 4/4. Meter therefore
  comes out of form rather than going into it, and `--meter` still overrides.
- **Notation.** A crooked bar gets its own time signature rather than being
  padded out with a rest, and it is placed at the position that splits the
  fewest notes — the same judgement a transcriber makes by ear.
- **Repeats.** The part *order* is a small closed set (`AB`, `ABC`); how many
  times each part is played is not. Form is scored on the run-length encoding of
  the labels, so `AABB`, `AABBB` and `AAABBAABB` all read as the same form,
  while the run of four that signals a section length wrong by a factor of two
  is still penalised.

AABB with whole bars still gets a prior bonus, because it is genuinely far more
common. Crooked stays reachable, at a mild disadvantage.

## Three-finger banjo

`fiddle-transcribe banjo` arranges a Scruggs-style part from any melody. It is an
*arranger*, not a transcriber, which is why it is useful today: point it at a
hand-corrected ABC and it produces a playable part regardless of how melody
extraction is doing.

The method mirrors how the style actually works. A roll pattern supplies a
string for every eighth-note slot; melody notes override the roll's choice
wherever they fall; remaining slots are filled with a chord tone on the roll's
string. Output is banjo tab, which is what players read.

Two details that generated banjo parts usually get wrong, and this one does not:

- **The fifth string stays an open drone** and never carries melody. Using it
  melodically is the single most obvious tell of a fake banjo part.
- **The octave shift into banjo range is chosen once for the whole tune**, not
  per note. Folding each note independently keeps everything technically in
  range while turning stepwise motion into octave leaps.

Capo follows real practice: 0 for G, 2 for A, 5 for C; D is played out of open
position rather than at the capo 7 the arithmetic would suggest.

## Findings on the synthetic corpus

Measured on the synthetic corpus (5 tunes × 3 difficulty levels, exact ground
truth). Full numbers in [`reports/`](reports/).

### Ablation: which design decisions actually pay?

Each column disables one thing. Measured twice, on two code states, with the
same conclusions — so these are not artifacts of a transient bug.

```
metric                     full   no-consensus  no-cleaning  no-onset-split    pyin
pitch_accuracy            0.663      0.645        0.694          0.553        0.855
onset_accuracy            0.375      0.379        0.448          0.302        0.557
duration_accuracy         0.344      0.335        0.371          0.412        0.477
key_accuracy              0.867      0.867        0.933          0.933        1.000
form_accuracy             0.467      0.467        0.533          0.400        0.467
confidence_auc            0.603      0.555        0.587          0.686        0.496
```

Three results cut against the design, and they are the most valuable output of
this work so far:

**Consensus contributes almost nothing** — 0.663 with it, 0.645 without. It is
the stated differentiating feature of the project. The likely cause is
mechanical rather than fundamental: consensus only ever sees what form detection
hands it, and form is correct less than half the time, so a gated feature sits
downstream of a coin flip. That is a hypothesis to test by conditioning the
measurement on form being correct — not an excuse.

**Pitch cleaning is net-harmful.** Disabling octave correction, median
filtering, vibrato smoothing, the jump gate and gap filling *improves* pitch,
onset, key and form accuracy simultaneously. Every one of those stages was
written on plausible reasoning about what ought to help. The per-stage switches
exist precisely so that reasoning could be checked, and it did not survive.
The next step is to find which specific stage is harmful rather than deleting
the ensemble.

**pYIN beats Essentia decisively** — 0.855 against 0.663 on pitch, and perfect
key accuracy. pYIN is a *monophonic* tracker, in principle the wrong tool for
polyphonic audio, included only as an honest baseline. That it wins challenges
the central technology choice.

**That last result is wrong about real audio, and measuring it was the single
most valuable thing in this project.** Run on the five real recordings, pYIN
produces essentially nothing:

| backend | synthetic pitch accuracy | real: voiced frames | real: notes found |
|---|---|---|---|
| Essentia Melodia | 0.663 | **46–54%** | 390–863 |
| pYIN | **0.855** | **0–2%** | **3–33** |

Three notes in a three-and-a-half minute recording. pYIN's voicing detector
never crosses threshold in a dense mix; the synthetic corpus places the fiddle
loudest with clean harmonics, which is exactly the condition a monophonic
tracker needs.

So **the synthetic corpus does not merely flatter the pipeline — it inverts the
ranking on the most consequential design decision in the system.** Anyone using
it to choose a melody backend would pick the one that yields three notes per
tune. This is the corpus's most important known limitation, and it generalises:
treat synthetic results as regression protection, never as evidence for a
design choice, without a real-audio check.

It also vindicates the spec's original instinct. Predominant-melody extraction
really is the right family of algorithm for this problem; the failures on real
audio are elsewhere.

**Confidence AUC sits between 0.50 and 0.69 in every variant.** The product
claim rests entirely on that number, and nothing in the current design moves it.

### What does work

- **Beat tracking is excellent on this material.** Section boundaries land
  within 0.02 beats of a true 32, and tempo error is under 1%. Old-time's steady
  guitar-and-bass pulse is a gift.
- **Key inference works, including modal tunes.** Mixolydian is detected as
  mixolydian and given its parent major's key signature.
- **Octave errors are essentially absent** (rate ~0.01) despite the synthetic
  banjo deliberately doubling the melody an octave down.

### What does not

- **Form detection is the bottleneck**, at 0.467 on synthetic audio and total
  failure on real audio. It gates consensus, so it caps the project's central
  idea before that idea gets a chance.
- **Confidence is not calibrated**, and the correction workflow depends on it.

**Verdict on the central hypothesis.** Not established. On synthetic audio the
domain-specific machinery is measurably not carrying its weight; on real audio
it has not been reached at all, because the contour going into it does not
contain the melody. Three structural findings, each of which cost a measurement:

1. **Similarity-based form analysis is completely flat in offset.** Shifting
   every block by the same amount leaves all pairwise similarities unchanged, so
   clustering finds the *period* of repetition but never its *phase*.
2. **A half-section clusters exactly as cleanly as a whole one.** No
   similarity-based objective separates them; the domain prior that sections
   repeat in runs of exactly two does.
3. **Exact slot matching is too brittle for human performances.** A ±1 slot
   tolerance roughly doubled the separation between same-section and
   different-section pairs.
4. **A section length incommensurate with the tune's period manufactures
   alternating structure.** Its phase slides forward each block, so even lags
   come back into phase and odd ones do not, and clustering reads that as an A/B
   contrast — at several times the cluster quality the *correct* length reaches,
   where everything is in phase and so everything looks alike. This was the real
   content of every `AAB…` result previously reported on real audio.

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
