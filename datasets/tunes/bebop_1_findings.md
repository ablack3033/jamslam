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

**4. The control was invalid, and this is a correction.** An earlier version of
this document claimed the same check reports 0% bass coupling on synthetic audio
containing a known fiddle melody, and called that a clean separation. It was an
artifact: the synthetic bass is too short-decayed for Melodia to track at all,
so there were *zero* simultaneously-voiced frames and the measure returned its
default. The check never ran. It has since been changed to return "not assessed"
rather than zero, so it fails closed.

The measure was also reformulated. It used to ask whether the melody-to-bass
interval was an octave or a fifth, which is confounded: a fiddle playing in its
low register over a root-position accompaniment genuinely sits an octave above
the bass much of the time. It now asks whether the interval is **locked**, since
a real melody's interval to the bass varies as the melody moves. On constructed
contours a rigid octave scores 1.00 and an independent melody 0.20; bebop_1
scores 0.63 under Melodia and 0.41 under the fiddle-specific extractor.

## What was genuinely learned

The recording *does* repeat. The pitch contour is self-similar at exactly 8.55s,
with harmonics at 2x through 6x -- 16, 32, 48, 64, 80 and 96 beats at the
detected 112.3 BPM. That is real structure, and it is what the accompaniment's
chord cycle looks like. **Repetition alone does not prove a melody was found**,
which is a limitation of the repetition objective worth remembering.

## Correction: the fiddle is present

The recordings are confirmed fiddle-led. An earlier reading of this evidence --
that no fiddle melody was recoverable -- was wrong. What the evidence actually
shows is that **Melodia's melody *selection* picks the wrong voice**, not that
the fiddle is missing.

Looking at every salience peak rather than Melodia's single choice makes this
visible. Over 90-114s there are three layers: a static line near A3, a static
line at D4, and a **third layer around F#4-D5 that moves melodically**. Melodia
selects the static D4 because it is the most salient; the moving line above it
is the fiddle.

That motivated a fiddle-specific extractor (`--melody-backend fiddle`) which
reuses Essentia's salience and contour tracking but replaces the final selection
with three domain rules: a melody moves where a drone does not, the melody is
usually the top voice, and a contour at a locked interval above a lower one is
its harmonic. On bebop_1 it moves time-on-a-single-pitch from 42% to 27% and
interval locking from 63% to 41%.

Not solved, but no longer mysterious: the melody is in the signal, and the task
is to select it.
