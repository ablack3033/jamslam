# Next steps: identify the tune, then adapt it

The pipeline is built, measured, and produces bad transcriptions. This document
says why, and what to do about it. It supersedes the "next steps" list in
`RESEARCH.md`.

The direction is **identify the tune against a catalog, then adapt the catalog
setting to the recording**. That is a different problem from the one the
pipeline currently attempts, and a much better posed one.

---

## The two measurements that set the direction

### 1. The melody collapse happens at thresholding, not at selection

Measured on `memory_of_home`. Pitch-class entropy and the share of energy in the
two most common pitch classes, at each stage of Basic Pitch's output:

| stage | pc-entropy | top-2 pitch classes | energy retained |
|---|---|---|---|
| raw posteriorgram | **3.55 bits** | **24.7%** | 100% |
| thresholded at 0.3 | 1.83 | 83.4% | 9.8% |
| thresholded at 0.5 (near default) | 1.58 | 86.7% | **2.9%** |
| after melody selection | 1.55 | 90% | — |
| *real old-time tunes, for scale* | *2.40* | *53%* | — |

Basic Pitch emits per-frame activations and only then thresholds them into
discrete note events. **Thresholding discards 97% of the energy**, and what
survives is the loudest, most sustained content — the guitar and banjo on the
tonic and the fifth. Everything downstream then inherits a line that is 86% two
pitch classes, against 53% for an actual tune.

This kills one plausible fix and enables another:

* **Reranking detected notes with a melodic prior cannot work.** The notes are
  not in the candidate pool. Measured: a stepwise melodic prior applied to the
  detected events made concentration *worse* (90% → 91%).
* **Matching against the un-thresholded posteriorgram can.** The information is
  still there at that stage.

### 2. ...but the fiddle is rarely the strongest bin in any single frame

Same recording: taking the per-frame argmax over the melodic register, only
**12.2%** of voiced frames peak on something other than the two dominant pitch
classes.

So the fiddle is present in the posteriorgram but weak. No frame-by-frame rule
will find it, because frame-by-frame it usually loses. What *can* find it is a
method that integrates weak evidence across a whole tune — which is exactly what
matching a known melody template does, and exactly what selection does not.

**Stated honestly:** high posteriorgram entropy proves the signal is not
collapsed. It does *not* prove the fiddle melody specifically is recoverable —
that energy includes every instrument plus room noise. The 12.2% figure is the
warning attached to the good news.

---

## The proposed architecture

```
recording
   │
   ├─ 1. IDENTIFY ....... match the un-thresholded posteriorgram against every
   │                      catalog tune × 12 transpositions × a tempo range.
   │                      Output: a ranked list with calibrated confidence.
   │
   ├─ 2. ALIGN .......... DTW the winning setting against the recording to find
   │                      where each pass of each section actually falls.
   │
   └─ 3. ADAPT .......... use the aligned setting as a backbone and let the
                          recording override it where the evidence is strong:
                          crooked bars, added/dropped beats, this player's
                          variations, the actual number of repeats.
```

Step 3 is the user's point and it matters: once the tune is known, the remaining
job is small and well-conditioned. We are no longer asking "what notes are
these?" but "where does this performance differ from the known setting?" — a
question with a strong prior and a much smaller answer space.

It also lowers the bar on step 1. Identification only has to be right about
*which tune*, not about every note.

---

## Why this is not simply the retrieval experiment that already failed

Retrieval was tried and did not discriminate: every catalog tune scored between
0.486 and 0.518, and the tentatively-correct answer was not in the top eight.
That attempt matched **beat-synchronous chroma** against catalog templates.

Three things are different now:

1. **The feature.** Chroma folds all octaves together and is dominated by the
   accompaniment's harmony. Old-time tunes share I-IV-V harmony and near
   identical diatonic content, so chroma cannot separate them — the failure was
   predictable in hindsight. A posteriorgram keeps register and time resolution,
   so a melody is matched *as a melody*.
2. **The catalog.** 20 hand-entered tunes is too few for a match to mean
   anything, and too few to contain the answer.
3. **The scoring.** The earlier version compared raw scores between tunes; a
   per-tune null model is now in place (`identify.py`), so scores are
   z-normalised and comparable.

None of this guarantees success. A cousin of this approach has already failed
once, and that should be held in mind.

---

## The catalog problem

This is the binding constraint, and it is practical rather than technical.

**thesession.org's database cannot be used.** 55,093 settings of 21,525 tunes,
with genuine old-time overlap, but its license explicitly prohibits processing
the material with Large Language Models, with an exception only for
accessibility work. It was downloaded, the license was read, and it was deleted
unused. Do not build on it.

**Public-domain sources are unreachable from this environment.** Ryan's Mammoth
Collection (1883) is properly public domain and has been transcribed to ABC, but
its host returns 000/403 here, as do abcnotation.com, thesession.org and
ibiblio. Reachable: PyPI and `raw.githubusercontent.com`. Nothing else tested so
far.

**Therefore the catalog has to come from somewhere else.** In rough order of
value:

1. **The library these recordings came from.** They are known to come from a
   small online collection. Naming it is worth more than any amount of
   inference, because it bounds the search to a few hundred candidates and
   probably *contains the answers*.
2. **A permissively-licensed or public-domain ABC corpus** mirrored on GitHub,
   which is reachable. Needs finding and its license checking.
3. **Hand-entered tunes**, as `src/fiddle/catalog/oldtime_starter.abc` already
   does for 20. Reliable but slow, and transcriptions written from memory should
   be treated as unverified until checked against a source.

---

## Ground truth is still the thing that unblocks everything

Every number in this repository is a proxy. Twice, a result that looked clean
turned out to rest on a control that never ran, and once a "strong" form result
turned out to be an artifact of the melody having collapsed to a single note.

The cheapest fix is unchanged: **the names of the tunes**, or **eight bars of
one transcribed by ear**. Either converts the whole proxy stack into a
measurement. Identification, if it works, produces this automatically — which is
the strongest argument for doing it first.

---

## Concrete order of work

1. **Build the catalog.** Source it (see above), and make it easy to extend:
   a documented directory of ABC, a validator, and a `catalog` CLI command that
   reports size and coverage.
2. **Match against the posteriorgram.** Score each tune × transposition ×
   tempo, using the un-thresholded activations rather than note events. Report
   ranked candidates with z-scores against the existing per-tune null model.
3. **Validate identification honestly.** A match is only believable if the
   scoring separates a held-out true tune from the rest of the catalog on
   synthetic audio *first*, where the answer is known.
4. **Align and adapt.** DTW the winning setting to the recording; emit the
   catalog melody as a backbone with per-bar deviations where the recording
   disagrees confidently.
5. **Only then revisit extraction.** If identification works, transcription
   quality stops being the gate. If it does not, the next lever is source
   separation (Demucs), still untested because its weights are unreachable here
   — untested, not disproven.

---

## Why form detection fails on three of the five recordings

Form succeeds on `bebop_2` and `bebop_5` and fails on `bebop_1`, `bebop_4` and
`memory_of_home`. It is not a form-detection bug. The extracted line on those
three does not contain distinguishable sections, and the evidence says so from
three independent directions.

**1. Everything above the quality floor is an artifact.** Searching section
lengths without the period constraint, the best cluster quality on each failing
recording occurs at a length *incommensurate with its own measured period* — 44
beats against a 16-beat period, 38 against 32, 28 against 48 — and every one of
them labels the blocks `ABABABAB…`. That is the phase-rotation artifact
documented above, not structure. At lengths that *are* commensurate, quality is
0.02–0.04 against a floor of 0.08.

This matters because form can be made to "succeed" on all three by lowering the
floor one line. It would emit `ABABABAB` at 44 beats for `bebop_1`. That is
manufacturing form, not finding it.

**2. Time tolerance does not help.** The note-based `slot_similarity` gained a
±1 slot tolerance long ago and it roughly doubled same-versus-different
separation, but the contour path — which is what actually runs — never got it.
Adding it changes cluster quality from 0.089 to 0.091. Not the problem.

**3. The lines are too pitch-concentrated for sections to differ.** Share of the
contour within one semitone of its own median, and the interquartile range:

| recording | within ±1 semitone | IQR | form |
|---|---|---|---|
| bebop_2 | **24%** | **12.0** | AABBC ✓ |
| bebop_5 | 34% | 5.1 | AABC ✓ |
| bebop_4 | 31% | 7.0 | fails |
| bebop_1 | **39%** | **4.0** | fails |
| memory_of_home | **42%** | 7.3 | fails |

`bebop_1` has half its line inside a four-semitone band. A fiddle tune spans an
octave or more. Per-block median pitch on that recording reads
`69 69 69 69 69 69 69 69 69 69 70 69…` across thirty-nine consecutive blocks;
on `bebop_2`, which works, it reads `69 76 64 64 70 76 67 67…`.

That last table also rules out the obvious domain fix. Old-time players call the
two parts the low part and the high part, so section register is real evidence
and it is independent of frame-wise similarity — but it needs register variation
to exist, and on the failing recordings there is almost none to read.

**Conclusion: this is the melody-extraction problem one level down, not a
separate one.** Form detection is already extracting everything that is there.

### Fixed on the way

The repetition period's integer tolerance was absolute (0.25 beats) rather than
relative. A measured 48.4-beat period is 0.8% from a whole number and was being
discarded, which dropped the tune's own period and sent form detection into an
unconstrained scan — the exact condition that produces artifacts. The tolerance
now scales with the period. `memory_of_home` keeps its 48-beat period; its form
still fails, for the reasons above.

---

## Tried and rejected: following one voice

Lowering the threshold left the extracted line spread across *more* pitch
classes than a real tune contains — 41% of notes in the top two against 53% for
a real tune — which is a line switching between instruments rather than a
melody. The obvious fix is to stop choosing per instant and select a whole path:
`fiddle/melody/voice.py` scores paths through the note events (reward for loud,
long notes; costs for movement, octave jumps, silence and overlap) and finds the
global optimum by dynamic programming.

It works as designed. It produces one connected line, and on real audio it
improves both proxies: concentration 88% → 79% and bass-locking 47% → 38% on
`memory_of_home`. **Against ground truth it is clearly worse, and it is not
shipped.**

| buried tier | pitch | pclass | onset | key | form | octave err | wrong/tune |
|---|---|---|---|---|---|---|---|
| loudest per instant | **0.86** | **0.93** | **0.85** | **1.00** | **1.00** | **0.070** | **10.2** |
| voice path | 0.67 | 0.77 | 0.65 | 0.60 | 0.80 | 0.107 | 17.6 |

Four weight settings were tried, including much stronger octave penalties and a
register-band term. All were worse. One earlier version was worse than useless:
it charged 0.10 per semitone of movement while an entire note was worth 0.09, so
the optimal path was mathematically forbidden from moving and locked onto a
drone — *worse* concentration than the selector it replaced, 88% → 93%.

**Why the naive rule is hard to beat.** Not because per-instant choice is good,
but because of the register floor above it. Above D4 the fiddle usually *is* the
loudest thing even when the guitar dominates the full mix, so "loudest above the
floor" is far better posed than it sounds. A path-based selector trades that
reliable local evidence for continuity it has no way to judge — it knows a
melody should be connected, but not which connected line is the fiddle. That is
precisely the knowledge a catalog would supply, which is another argument for
doing identification before any more selector work.

### The corpus could not previously test this at all

Worth recording separately, because it nearly produced a false conclusion. Every
difficulty tier put the fiddle at gain 0.9–1.0 against accompaniment of at most
0.46, so **"keep the loudest note" was close to correct by construction** and the
corpus scored it above every alternative while being structurally incapable of
testing one.

A `buried` tier now exists: fiddle 0.55 against guitar 0.85 and banjo 0.80,
which is the condition the real recordings are actually in. The verdict above is
from that tier, so it means something. Any future melody-selection work should
be measured there.

It also exposes the next real problem. On `buried`, *both* selectors identify
notes reasonably (pitch 0.86, pitch-class 0.93, onset 0.85) and both produce
duration accuracy of **0.03**. Note identity survives a buried fiddle; note
*length* does not, which matches what the real recordings look like.

---

## Done: the threshold was lowered and re-measured

Basic Pitch's note-creation thresholds are now config (`melody.basicpitch_*`)
and the default moved from 0.5/0.3 to **0.3/0.15**.

**What the extra notes are.** Not noise. Median note duration holds near 190 ms
at every threshold and under 2% of notes are shorter than a sixteenth. Notes per
second times median duration gives average polyphony, and that is what moves:

| onset / frame | notes/sec | median duration | ⇒ simultaneous notes |
|---|---|---|---|
| 0.50 / 0.30 | 9.1 | 197 ms | 1.8 |
| 0.30 / 0.15 | 36.7 | 186 ms | 6.8 |
| 0.20 / 0.10 | 117.4 | 186 ms | 21.8 |
| 0.10 / 0.03 | 240.1 | 151 ms | 36.3 |

A jam has perhaps six to eight simultaneous pitches, so 0.3/0.15 is roughly
faithful and the lower settings are over-detecting.

**End to end against ground truth** on the synthetic corpus:

| thresholds | pitch | onset | dur | key | form | octave err | wrong/tune | AUC |
|---|---|---|---|---|---|---|---|---|
| 0.5 / 0.3 | 0.88 | 0.60 | **0.44** | 0.73 | **0.80** | 0.024 | 5.7 | 0.600 |
| **0.3 / 0.15** | **0.90** | **0.67** | 0.35 | **0.93** | 0.67 | **0.011** | **2.9** | 0.634 |
| 0.2 / 0.1 | 0.84 | 0.66 | 0.20 | **1.00** | 0.73 | 0.015 | 7.4 | **0.700** |

The fundamentals improve — key 0.73 → 0.93, octave errors halved, wrong notes
per tune halved — and structure degrades: duration 0.44 → 0.35, form 0.80 →
0.67. A denser pool makes the "loudest note above the floor" selector switch
voices more often, which chops note boundaries and blurs the section clustering
built on top of them.

That trade is worth taking **only because of where this is going**. Under
catalog matching, duration and form come from the matched setting; pitch and key
are what the match is scored on. If the catalog approach is abandoned, revisit
this default.

**The methodological finding is the more valuable one.** The extraction-level
proxies said lowering the threshold was harmful — repetition fell from 0.29 to
0.19 on a real recording — while ground-truth note accuracy rose. That is the
third time on this project a proxy has pointed the wrong way. The two reports
are kept side by side deliberately:
[`threshold-sweep-proxies.txt`](../reports/threshold-sweep-proxies.txt) and
[`threshold-sweep-groundtruth.txt`](../reports/threshold-sweep-groundtruth.txt).

It also sharpens the earlier conclusion that reranking cannot work. That was
measured on the *default* threshold's candidate pool. At 0.3/0.15 the pool is
much richer and its notes are of plausible duration, so a sufficiently strong
prior — a specific known melody, not a generic stepwise preference — has real
material to select from. Lowering the threshold does not fix melody selection;
it makes catalog matching more likely to work.
