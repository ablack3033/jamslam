# Gold standard: `memory_of_home`

This is the reference real-world recording. Every example in the documentation
runs against it, and it is the first thing to check after any algorithm change.

    recording.m4a     4m24s, mono 48 kHz AAC, phone recording of a jam
    diagnostics.png   the five-panel pipeline view for this recording
    meta.json         the pipeline's own output, recorded for tracking only
    expected.abc      MISSING - this is the one thing needed to make it scorable

## Why this recording is the gold standard

Not because it works. Because it is honest. It is a real phone recording of a
real circle, it is 4½ minutes long (so there are many passes of each section to
feed consensus), and **the pipeline currently fails on it**. That makes it the
most useful single file in the repository: a synthetic corpus can tell you that
your DSP is not broken, but only this tells you whether the product works.

## Current result

```
$ fiddle-transcribe datasets/tunes/memory_of_home/recording.m4a --plots --identify

key           A major  (confidence 0.52)
meter         4/4  (confidence 0.37)
tempo         103.4 BPM
form          ABB  (20 bars/section, confidence 0.35)
notes         380 canonical, 72 flagged uncertain
melody        essentia backend, 61% of frames voiced
```

Read that as a failure, and specifically these failures:

1. **Form is wrong.** `ABB` with "20 bars per section" is the search hitting its
   own upper bound, which is what it does when there is no structure to find.
   A real old-time tune is not 20 bars per section.
2. **The melody was not extracted.** In `diagnostics.png`, panel 3 (raw vs
   cleaned F0) shows a sparse, fragmented contour sitting on sustained low
   pitches, and panel 5 shows a long run of one low note. The extractor spent
   much of the recording tracking **drones and accompaniment, not the fiddle**.
3. **Identification finds nothing**, which independently confirms (2) --
   see below.

## What this recording taught us

Diagnosing it produced two real bug fixes and one structural finding.

**`min_frequency` was far too low.** It was 130 Hz (~C3), set deliberately below
the fiddle so that octave errors would still be visible to the cleaner. On real
jam audio that was a serious mistake: guitar, bass and banjo own the 100–250 Hz
band and are usually closer to the phone than the fiddle is, so Melodia's
salience function tracked them. Raising the floor to 250 Hz moves the median
extracted pitch from MIDI 57 (A3, an accompaniment drone) to MIDI 69 (A4, the
fiddle's register). That is now the default.

**Identification needed IDF weighting.** Before it, the matcher ranked the same
tune first for all five test recordings *at an identical score* — matching only
on the all-zeros interval pattern that repeated notes produce. Weighting n-grams
by rarity fixed it. The regression test is
`test_repeated_notes_alone_do_not_identify_anything`.

**Failure to identify is a useful signal.** The matcher scores a correct melody
at 1.000 against its own catalog entry, and still at 0.37 with one note in seven
corrupted. This recording scores 0.15. That gap is not ambiguity about which
tune it is — it is evidence that the transcription does not contain a melody.

## What is needed next

`expected.abc`. Without ground truth this recording can be inspected but not
scored, so it cannot drive the metrics.

Two ways to produce it, in order of preference:

1. **Name the tune.** If you know what it is, drop the setting in as
   `expected.abc` — from a tunebook, from thesession.org, or typed from memory.
   Any ABC works; the parser validates bar counts and will refuse a typo.
2. **Correct a transcription.** Run the tool, open the emitted `.abc`, fix it by
   ear, and save it here. This is the intended loop, but it only saves effort
   once transcription quality is good enough to be worth correcting — which for
   this recording it is not yet.

Once `expected.abc` exists:

```
fiddle-transcribe eval --dataset datasets/tunes
```

## Reproducing the analysis

```
# Full transcription with diagnostics and catalog matching
fiddle-transcribe datasets/tunes/memory_of_home/recording.m4a --plots --identify

# Just the identification, reusing an existing transcription
fiddle-transcribe identify --from-json recording.notes.json

# Match against a real tune library instead of the small builtin catalog
fiddle-transcribe identify recording.m4a --catalog /path/to/abc/library
```
