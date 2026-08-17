# bebop_1: what the investigation found

`bebop_1` was chosen as the focus recording because it is the one tentatively
identified (possibly Sandy River Belle) and therefore the closest to having
usable ground truth.

## Conclusion

**The extracted contour is the accompaniment, not the fiddle.** No amount of
work on segmentation, form detection or consensus fixes that, because the
melody is not in the signal those stages receive.

## The evidence

**1. The contour is locked to the bass.** Extracting a second contour restricted
to the bass register (70-260 Hz) and comparing:

| interval between "melody" and bass | share of frames |
|---|---|
| exactly one octave | 63.5% |
| a twelfth | 10.7% |
| unison / fifth / two octaves | 10.2% |
| **any fixed harmonic interval** | **84%** |

A real melody wanders relative to the bass line even when both are built from
chord tones. A harmonic of the bass does not.

**2. The contour sits on one pitch.** Over 90-114s it is parked on D4 (MIDI 62)
with brief excursions to F#4 and A4 -- the D major triad. The bass contour over
the same window alternates between D3 (50) and A3 (57). The "melody" is the
accompaniment root, doubled.

**3. There is nothing in the fiddle's register.** Raising the extractor's
frequency floor:

| floor | voiced | time on the single commonest pitch |
|---|---|---|
| 220 Hz | 50% | 42% on D4 |
| 300 Hz | 25% | 46% on A4 |
| 400 Hz | 15% | **76% on A4**, interquartile range 0.3 semitones |
| 500 Hz | 1% | -- |

Above 400 Hz there is no melody, only a sustained A4 drone. An old-time fiddle
tune in D lives around D5-A5; that region is empty here.

**4. The control passes.** The same check on synthetic audio containing a known
fiddle melody -- including the hardest setting, with two fiddles, banjo, guitar
and bass -- reports **0% bass coupling** and is not flagged. The measure
separates the two cases completely.

## What was genuinely learned

The recording *does* repeat. The pitch contour is self-similar at exactly 8.55s,
with harmonics at 2x through 6x -- 16, 32, 48, 64, 80 and 96 beats at the
detected 112.3 BPM. That is real structure, and it is what the accompaniment's
chord cycle looks like. **Repetition alone does not prove a melody was found**,
which is a limitation of the repetition objective worth remembering.

## What this cannot settle

Whether the fiddle is absent, merely quiet, or playing in the low register where
the guitar masks it. That needs someone who can listen. Two things would resolve
it quickly:

1. **Is bebop_1 fiddle-led?** If it is banjo- or guitar-led, the premise of the
   whole pipeline does not apply to it and it is the wrong test case.
2. **A recording where the fiddle is close to the microphone.** The melody has
   to win the salience contest against instruments that are physically nearer
   the phone, and in these five recordings it does not.
