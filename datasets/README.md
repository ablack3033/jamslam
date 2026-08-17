# Evaluation corpus

Two kinds of corpus, scored by identical code (`fiddle.eval.metrics`).

## `tunes/` — real recordings (you supply these)

This is the corpus that answers the actual question. Layout, one directory per
recording:

```
datasets/tunes/
    soldiers_joy_fri_jam/
        recording.m4a       # or .wav / .mp3 / .flac / .ogg
        expected.abc        # ground truth (or expected.musicxml)
        meta.json           # optional
```

`meta.json` is optional and looks like:

```json
{"key": "D major", "meter": "4/4", "tempo_bpm": 124, "form": "AABB"}
```

Without it the note-level metrics still work; the key/tempo/meter checks are
skipped and the run is annotated as such.

Run it with:

```
fiddle-transcribe eval --dataset datasets/tunes
```

### Writing ground truth

ABC, not MusicXML. An A part is one line you can type from memory and proofread
at a glance; the equivalent MusicXML is hundreds of lines nobody will check. The
parser **validates that every section contains a whole number of bars** and
raises if not, so a typo fails loudly instead of silently corrupting the corpus.

The fastest way to produce one: transcribe the recording, correct the handful of
flagged notes in the emitted `.abc`, and save it as `expected.abc`.

```
X:1
T:Soldier's Joy
M:4/4
L:1/8
K:D
P:A
a2 fa gfef|dcde fdcd|a2 fa gfeg|fdcd e2 de|
a2 fa gfef|dcde fdcd|a2 fa gfeg|fdcd e2 d2|
P:B
f2 df e2 ce|dcde fdcd|f2 df e2 ce|fdcd e2 de|
f2 df e2 ce|dcde fdcd|a2 fa gfeg|fdcd e2 d2|
```

Ground truth is the **canonical tune**, not a literal transcription of the
performance. Do not notate ornaments, slides, or bowing — the system is being
scored on whether it recovers the tune, and encoding performance detail in the
truth would penalise it for succeeding.

## Synthetic corpus (generated, exact ground truth)

Rendered on demand from `fiddle/corpus/library.py`, so it is reproducible and
costs no repository space:

```
fiddle-transcribe eval                      # score it
fiddle-transcribe build-corpus datasets/synth   # write the audio to disk
```

Three difficulty levels per tune:

| level | contents |
|---|---|
| `clean` | solo fiddle, slight vibrato and jitter — the control condition |
| `jam` | fiddle, guitar, banjo, bass, mandolin, room noise, drones, tempo drift |
| `hard` | two fiddles, banjo forward, heavy droning, sloppy time, more noise |

**Synthetic success is necessary, not sufficient — and on some questions it is
actively misleading.** It proves the engine is not broken and catches
regressions cheaply. It cannot tell you whether the system works on a phone
recording of a real circle, because it shares none of the acoustics that make
that hard.

> **Do not use this corpus to choose a melody backend.** Measured here, pYIN
> beats Essentia by a wide margin (0.855 against 0.663 pitch accuracy). Measured
> on real recordings, pYIN finds 0–2% voiced frames and as few as three notes in
> a three-minute tune, while Essentia finds 46–54%. The corpus mixes the fiddle
> loudest with clean harmonics, which is precisely the condition a monophonic
> tracker needs and a jam does not provide. The ranking is not merely optimistic
> here, it is inverted.

The general rule this implies: use synthetic results as regression protection,
and require a real-audio check before acting on them as evidence for a design
decision.
