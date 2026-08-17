# Research: making fiddle-jam transcription actually work

Constraints taken as given: iPhone-app recordings, noisy and low quality; always
old-time fiddle tunes; the fiddle always leads; melody only, as a first step; and
the interface must stay **recording in, sheet music out**.

The established problem is not pitch tracking. It is **melody selection**: on a
phone recording the guitar and bass are often physically closer to the mic than
the fiddle, so a generic "predominant melody" chooses them. Measured on
`bebop_1`, Melodia's output holds a single fixed interval to the bass in 63.5% of
frames.

Two measures are used throughout, neither needing ground truth:

* **locked** — share of frames holding one fixed interval to a bass-register
  contour. Lower is better; a real melody's interval to the bass varies.
* **REP** — excess self-similarity at the tune's period. Higher is better, but it
  rewards *whatever* repeats, and a chord cycle repeats more reliably than a
  fiddler, so it must never be read alone.

Everything below is measured on `bebop_1`, 5m36s of real jam audio.

---

## Result: Basic Pitch with a register floor

| approach | voiced | modal | locked ↓ | REP ↑ |
|---|---|---|---|---|
| Melodia (baseline) | 0.50 | 42.1% | 63.5% | 0.287 |
| Melodia + sustain mask | 0.41 | 35.0% | 53.0% | 0.248 |
| frame-level Viterbi | 0.97 | 11.1% | 8.7% | 0.041 |
| contour-level DP | 0.53 | 39.7% | 24.4% | 0.173 |
| Basic Pitch + sustain mask | 0.38 | 40.0% | 28.3% | 0.284 |
| **Basic Pitch, floor D4, loudest** | **0.52** | 40.7% | **26.5%** | **0.286** |

Basic Pitch is the first approach that improves bass independence **without**
sacrificing repetition — it more than halves locking while nudging REP up. Every
other candidate traded one against the other, which is why none of them could be
called an improvement.

**Why it works.** Melodia produces a salience function and then commits to one
line; selection happens inside the algorithm, on generic criteria. Basic Pitch
produces **discrete note events for every voice**, and hands the selection
problem to us intact. Selecting among notes is far better posed than selecting
among salience peaks: notes have register, duration, amplitude and overlap, all
of which can be reasoned about directly. Its layer separation on `bebop_1` is
unambiguous:

    D2(78) G2(23)      bass
    A3(34) B3(28)      mid / accompaniment
    D4(123) F#4(50) A4(74) B4(34)   melodic register

The selection rule that measured best is deliberately simple: **discard
everything below D4, then take the loudest note sounding at each instant.**

**This is now the default backend.** The numbers above were produced by an
ad-hoc script, so before defaulting to it the *shipped* code path was run
end-to-end in the isolated environment and reproduced them: voiced 0.52,
median 69.0, modal 40.7%, locked 26.5%, REP 0.284 against the script's 0.286.

**Caveats.** Basic Pitch pins `numpy<2` and pulls TensorFlow, so it cannot share
the main environment — it needs its own venv. Because the default therefore
fails on a plain checkout, backend resolution falls back to Essentia and prints
a warning; the fallback is loud on purpose, since a silent one would let someone
believe they were running Basic Pitch when they were not.

Its model ships inside the wheel, so it needs no network at inference, which
also makes it viable for the offline/WASM deployment discussed elsewhere.

---

## What was tried and rejected

**Source separation (Demucs).** Almost certainly the single biggest available
lever: removing the bass stem attacks the problem at its root. The package
installs, but the pretrained weights are fetched at runtime from hosts this
environment cannot reach. **Untested, not disproven** — this should be the first
thing tried in an environment with network access.

**Tune retrieval instead of transcription.** Tempting given a bounded
repertoire: identify the tune from robust audio features and serve the known
setting, which yields clean notation immediately. Tested with beat-synchronous
chroma matched against catalog templates over all 12 transpositions. It does not
discriminate — every tune scored between 0.486 and 0.518, a 3% spread, and the
tentatively-correct answer was not in the top eight. Old-time tunes share I-IV-V
harmony and near-identical diatonic content, so chroma is too coarse a feature.
Retrieval is not dead as an idea, but it needs melodic features, which need the
transcription this was meant to avoid.

**pYIN.** A monophonic tracker, included as an honest baseline. Wins by a wide
margin on synthetic audio and finds 0-2% voiced frames on real recordings. Its
voicing detector never fires in a dense mix.

**Frame-level Viterbi.** Cured the symptoms and broke the patient: locking fell
to 8.7% while REP collapsed sevenfold. Aimless wandering also scores well on
locking. Frames are the wrong unit.

---

## Kept: the bowed/plucked asymmetry

In an old-time jam the fiddle is the **only bowed instrument** — banjo, guitar,
mandolin and bass are all plucked. Plucked notes decay from a sharp attack;
bowed notes sustain. Masking the spectrogram by how close each bin sits to its
own recent peak emphasises sustained sources:

| sustain exponent | modal | locked | REP |
|---|---|---|---|
| 0 (off) | 42.1% | 63.5% | 0.287 |
| 1 | 39.0% | 59.6% | 0.286 |
| 3 | 35.0% | 53.0% | 0.248 |

Modest, monotonic, costs nothing, and needs no model. It is a physical property
of the ensemble that generic melody extraction cannot know.

### ...but it does not combine with Basic Pitch

The obvious next step, since the two attack the problem independently. Tested:

| | notes | voiced | locked ↓ | REP ↑ |
|---|---|---|---|---|
| Basic Pitch alone | 3056 | 0.52 | **26.5%** | **0.286** |
| + sustain² | 2342 | 0.42 | 28.2% | 0.260 |
| + sustain³ | 1967 | 0.38 | 28.3% | 0.284 |
| + sustain⁴ | 1592 | 0.34 | 31.3% | 0.284 |

**The mask makes Basic Pitch slightly worse, monotonically.** Locking rises
26.5% → 31.3%, coverage falls 0.52 → 0.34, and repetition never improves. Note
count halves: the mask is deleting signal Basic Pitch was using.

They are **substitutes, not complements.** The mask helped Melodia because
Melodia had no other way to tell the fiddle from the guitar; Basic Pitch already
separates the voices, and does it better than a spectral heuristic can. Applying
both pays the mask's cost -- discarded signal -- for a discrimination that is
already handled.

Keep the mask for the Melodia path, where it earns its place. Do not stack it
under Basic Pitch.

---

## End-to-end on all five recordings

Full pipeline, not just extraction. Reports in
[`reports/real-backend-comparison.txt`](../reports/real-backend-comparison.txt).

| recording | Melodia form | Basic Pitch form | Melodia voiced | BP voiced |
|---|---|---|---|---|
| bebop_1 | **failed** | AABBCDD / 15 bar | 48% | 52% |
| bebop_2 | **failed** | AABCD / 12 bar | 54% | 58% |
| bebop_4 | **failed** | AABBCCD / 16 bar | 40% | 65% |
| bebop_5 | **failed** | AABBCD / 16 bar | 47% | 53% |
| memory_of_home | ABBCD / 20 bar | AABBCD / 16 bar | 46% | 77% |

Three things moved together, which is the first time that has happened here.

**The melody-presence check flips completely.** Under Melodia all five were
flagged, at 75-91% of frames locked to the bass. Under Basic Pitch **all five
pass**. That check was written to be hard to satisfy and has never before
returned a clean verdict on real audio.

**Form detection now returns structure on every recording** rather than failing
outright on four of five. Every result still begins `AAB…`, which is the right
shape, and confidences are 0.11-0.34, which is honestly low.

**Identification strengthens sharply**: best z rises from 2.7 to 7.2. bebop_1
now clears the z threshold (7.2 against a bar of 6) but fails on margin (0.3),
and memory_of_home clears margin (4.4) but not z. **No recording is confidently
identified**, so nothing here is confirmation of a tune name.

### What is still wrong

The section lengths are implausible. 12 to 16 bars per section, with forms
running to six or seven distinct letters, is not what an old-time tune looks
like — AABB with 8-bar sections is. So form is finding *something* periodic and
carving it wrongly, rather than finding the tune's actual structure. **The next
section identifies what that something was**, and it is not a near miss.

Uncertain-note counts remain high (216-450 of 257-683 notes). At those rates the
promised experience — correct two to five flagged notes — is not close.

---

## Form: the phase-rotation artifact

Form detection used to return a section length and an `AAB…` label sequence for
every recording. Those results were not weak versions of the right answer; they
were an artifact, and naming it is the main form finding here.

Measured on `bebop_1`, mean block similarity as a function of block lag
(full table in [`reports/block-similarity-by-lag.txt`](../reports/block-similarity-by-lag.txt)):

| block length | lag 1 | lag 2 | lag 3 | lag 4 | spread |
|---|---|---|---|---|---|
| 16 beats | 0.525 | 0.492 | 0.523 | 0.524 | **0.04** |
| 32 beats | 0.504 | 0.537 | 0.497 | 0.504 | **0.05** |
| 36 beats | 0.202 | 0.426 | 0.223 | 0.494 | 0.29 |
| 44 beats | 0.245 | 0.385 | 0.224 | 0.514 | 0.33 |

The tune's period is 16 beats. At block lengths that are **multiples of it**,
every block starts at the same point in the tune, so all blocks resemble each
other equally — flat, and useless to a clustering algorithm. At lengths that are
**not**, each block's phase slides forward by a fixed amount, so blocks an even
number apart come back into phase and odd ones do not. Clustering reads that as
a clean A/B contrast and reports `ABABAB…`.

The artifact wins outright: cluster quality 0.21 at 36 beats against 0.01 at the
commensurate lengths. It is not a near miss to be down-weighted — a hypothesis
incommensurate with the period is **guaranteed** to manufacture alternating
structure, so those lengths are no longer generated at all. Widening the
allowed period ratios by a single beat was tried and put the artifact straight
back (quality 0.04, plainly-noise labels).

What is left once the artifact is removed is the real problem: **at the correct
16-beat length, similarity is 0.50 flat.** The A part and the B part resemble
each other exactly as much as two passes of the A part do. The extracted melody
carries the tune's periodicity but not enough detail to tell its sections apart.
That is a melody-quality problem, not a search problem, and no form prior can
fix it.

So form now declines rather than guessing. Of the five recordings, two return a
section and three report no structure found:

| recording | before | after |
|---|---|---|
| bebop_1 | AABBC / 9 bar | *no structure* (quality 0.012) |
| bebop_2 | AABCD / 12 bar | 8 bars of 4/4, one bar of 3 (quality 0.44) |
| bebop_4 | AABBCCD / 16 bar | *no structure* (quality 0.071) |
| bebop_5 | AABBCD / 16 bar | *no structure* |
| memory_of_home | AABBCD / 16 bar | 6 bars of 4/4 (quality 0.081) |

Three fewer scores get section structure, and that is the honest outcome: a
wrong split makes consensus average unrelated music together, which is worse
than not averaging at all.

On the synthetic corpus, where the melody *is* good enough to separate sections,
the same code moves form accuracy **0.67 → 0.80** (like-for-like, same
environment, same backend).

### What flexibility was added

Three constraints the repertoire actually has, none of which the search used:

* **Meter comes out of form, not into it.** Section length in beats is what the
  audio determines; 2/4 and 4/4 are metrically nested, so the same recording is
  correctly barred either way. Beat stress can barely choose between them. Bar
  *count* can, because the 8-bar section is close to universal: a 16-beat
  section is 8 bars only in 2/4, a 32-beat section only in 4/4.
* **Odd bars.** A section may carry one bar of a different length — a 2/4 bar
  dropped into a 4/4 tune is idiomatic here. Where that bar goes is decided by
  which placement cuts through the fewest notes.
* **Repeat counts vary.** The part *order* is a small closed set (`AB`, `ABC`);
  how many times each part is played is not. Scoring the run-length encoding
  rather than whole label strings makes `AABB`, `AABBB` and `AAABBAABB` all
  score as the same form, while still penalising the run of four that signals a
  section length wrong by a factor of two.

---

## The real blocker

**No ground truth.** Every measure above is a proxy, and proxies are gameable —
one of them was gamed by each of the failed attempts. Twice in this project a
result that looked clean turned out to rest on a control that never ran.

Basic Pitch is the first result where **both** proxies move the right way at
once, which is why it is reported as a finding rather than a hypothesis. But
"better than Melodia on two proxies" is still not "good enough to hand a fiddler
sheet music."

**Eight bars of `bebop_1` transcribed by ear, as ABC, would settle more than any
further algorithm work.** It converts every open question here into a
measurement, and it makes the whole regression corpus real.

---

## Recommended architecture

```
iPhone recording
      │
      ├─ Basic Pitch .......... discrete note events for every voice
      │                         (NOT preceded by the sustain mask -- measured
      │                          worse; they are substitutes, see above)
      │
      ├─ melody selection ..... register floor D4 + loudest voice
      │                         (+ diatonic prior from chroma; untested)
      │
      └─ existing pipeline .... rhythm → form → consensus → MusicXML
```

The sustain mask stays on the Melodia path, which remains the no-extra-install
default and where the mask does earn its place.

The downstream pipeline is unchanged and already consumes note events, so this
slots in behind the existing `MelodyExtractor` interface without touching
anything after it.

Next steps in priority order:

1. **Get ground truth for one recording.** Everything else is unfalsifiable
   without it.
2. **Try Demucs** in an environment with network access; remove the bass stem
   before extraction.
3. **Add the diatonic prior** to melody selection; chroma pins the root reliably
   even on noisy audio (`bebop_1` → D, scale fit 0.70). Untried.
4. ~~Combine the sustain mask with Basic Pitch~~ — tested, does not help; see
   above. They are substitutes rather than complements.
