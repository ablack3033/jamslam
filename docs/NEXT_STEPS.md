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
