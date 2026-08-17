"""Render a :class:`Tune` as notation via music21, and export MusicXML/MIDI.

Notation choices here follow one rule: print the tune, not the performance.
Vibrato, bow articulation and millisecond timing deviations are deliberately
absent -- they were measured upstream and used as evidence, and re-encoding
them would produce a score no fiddler wants to read.

Repeated sections are written with repeat barlines rather than duplicated, both
because that is how the tradition notates them and because it makes the form we
inferred visible on the page.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .domain import Note, Tune

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]

# Notes scoring below this get a visual marker, so a human correcting the
# transcription knows where to look first.
UNCERTAIN_COLOR = "#cc0000"


def tune_to_stream(tune: Tune, mark_uncertain_below: float = 0.7):
    """Build a music21 Score from a Tune."""
    from music21 import bar, clef, key, meter, note as m21note, stream, tempo

    score = stream.Score()
    part = stream.Part()
    part.insert(0, clef.TrebleClef())

    sharps = tune.analysis.get("key_sharps", 0)
    part.insert(0, key.KeySignature(sharps))
    part.insert(0, meter.TimeSignature(tune.meter))
    part.insert(0, tempo.MetronomeMark(number=round(tune.tempo, 1)))
    if tune.title:
        score.insert(0, _metadata(tune.title))

    beats_per_bar = _beats_per_bar(tune.meter)
    measure_number = 1

    current_meter = beats_per_bar
    for section in tune.sections:
        m21_measures = []
        bar_start = Fraction(0)
        for measure in section.measures:
            m = stream.Measure(number=measure_number)
            measure_number += 1
            offset = Fraction(0)
            # A crooked bar carries its own time signature rather than being
            # padded out to the prevailing meter. Padding a 3-beat bar with a
            # rest would misrepresent the tune: crooked bars are how it is
            # actually played, not a shortfall to be filled.
            bar_beats = _measure_length(measure, beats_per_bar)
            if bar_beats != current_meter:
                m.insert(0.0, meter.TimeSignature(_meter_string(bar_beats)))
                current_meter = bar_beats
            for n in measure.notes:
                local = n.start_beats - bar_start
                if local > offset:
                    # A gap inside the bar is a rest we did not explicitly emit.
                    r = m21note.Rest(quarterLength=float(local - offset))
                    m.insert(float(offset), r)
                    offset = local
                el = _make_element(n, sharps, mark_uncertain_below)
                m.insert(float(local), el)
                offset = local + n.duration_beats
            if offset < bar_beats:
                m.insert(float(offset),
                         m21note.Rest(quarterLength=float(bar_beats - offset)))
            m21_measures.append(m)
            bar_start += bar_beats

        if m21_measures and section.repeats >= 2:
            m21_measures[0].leftBarline = bar.Repeat(direction="start")
            m21_measures[-1].rightBarline = bar.Repeat(direction="end")
        for m in m21_measures:
            part.append(m)

    score.insert(0, part)
    return score


def _make_element(n: Note, sharps: int, threshold: float):
    from music21 import note as m21note

    ql = float(n.duration_beats)
    if n.is_rest:
        return m21note.Rest(quarterLength=ql)
    el = m21note.Note(_spell(n.pitch, sharps), quarterLength=ql)
    if n.confidence < threshold:
        el.style.color = UNCERTAIN_COLOR
        # Editorial data survives into MusicXML, so a UI can read it back.
        el.editorial.confidence = round(n.confidence, 3)
        if n.alternatives:
            el.editorial.alternatives = [
                {"pitch": a.pitch, "weight": round(a.weight, 3)} for a in n.alternatives
            ]
    return el


def _spell(midi: int, sharps: int) -> str:
    """Choose an enharmonic spelling consistent with the key signature.

    Sharp keys get sharp spellings and flat keys flat ones. This is crude
    compared with proper spelling by harmonic function, but for the diatonic
    material old-time tunes are built from it is correct essentially always, and
    it never produces the double accidentals a naive MIDI conversion can.
    """
    names = _SHARP_NAMES if sharps >= 0 else _FLAT_NAMES
    octave = midi // 12 - 1
    return f"{names[midi % 12]}{octave}"


def _measure_length(measure, default: Fraction) -> Fraction:
    """The bar's actual length in beats.

    Taken from the notes it contains rather than assumed, so a crooked bar is
    notated as the length it really is. Falls back to the prevailing meter for
    an empty bar, where there is nothing to measure.
    """
    if not measure.notes:
        return default
    span = max(n.end_beats for n in measure.notes) - min(
        n.start_beats for n in measure.notes
    )
    total = measure.total_beats
    length = max(total, span)
    # Only accept a crooked length if it is a clean simple value; anything else
    # is a segmentation artifact and the prevailing meter is the safer reading.
    for candidate in (Fraction(1), Fraction(2), Fraction(3), Fraction(4),
                      Fraction(5), Fraction(6), Fraction(7), Fraction(8)):
        if abs(float(length - candidate)) < 0.26:
            return candidate
    return default


def _meter_string(beats: Fraction) -> str:
    """Render a bar length in quarter-note beats as a time signature."""
    if beats.denominator == 1:
        return f"{beats.numerator}/4"
    if beats.denominator == 2:
        return f"{beats.numerator}/8"
    return "4/4"


def _beats_per_bar(meter_str: str) -> Fraction:
    num, den = meter_str.split("/")
    return Fraction(int(num) * 4, int(den))


def _metadata(title: str):
    from music21 import metadata

    md = metadata.Metadata()
    md.title = title
    md.composer = "traditional"
    return md


def export_musicxml(tune: Tune, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tune_to_stream(tune).write("musicxml", fp=str(path))
    return path


def export_midi(tune: Tune, path: str | Path) -> Path:
    """Write MIDI for playback only.

    MIDI is an output, never a source of truth: it cannot represent our
    confidence, alternatives, or inferred form. Anything the application needs
    to remember lives in the Tune and in the JSON export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tune_to_stream(tune).write("midi", fp=str(path))
    return path


#: Sharped letters by key signature, in the order sharps are added.
_SHARP_ORDER = "FCGDAEB"
#: Flatted letters, in the order flats are added.
_FLAT_ORDER = "BEADGCF"


def export_abc(tune: Tune, path: str | Path) -> Path:
    """ABC export, so a corrected transcription can become ground truth cheaply.

    Written by hand rather than through music21, which reads ABC but does not
    write it. This closes the loop that makes the regression corpus grow: run a
    transcription, fix the handful of flagged notes in the ``.abc``, drop it
    next to the recording as ``expected.abc``, and the tune is now a permanent
    regression test.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tune_to_abc(tune))
    return path


def tune_to_abc(tune: Tune) -> str:
    sharps = int(tune.analysis.get("key_sharps", 0))
    beats_per_bar = _beats_per_bar(tune.meter)
    root = tune.key.split()[0] if tune.key else "C"
    mode = tune.key.split()[1][:3].lower() if len(tune.key.split()) > 1 else "maj"

    lines = [
        "X:1",
        f"T:{tune.title or 'Untitled'}",
        f"M:{tune.meter}",
        "L:1/8",
        f"Q:1/4={round(tune.tempo)}",
        f"K:{root}{'' if mode == 'maj' else mode}",
    ]
    for section in tune.sections:
        lines.append(f"P:{section.name}")
        bars = []
        for measure in section.measures:
            tokens = [_abc_token(n, sharps) for n in measure.notes]
            # Pad a short bar with a rest so the barlines agree with the meter.
            # Without this, an incomplete final measure silently shifts every
            # later barline when the file is read back as ground truth.
            missing = beats_per_bar - measure.total_beats
            if missing > 0:
                tokens.append(f"z{_abc_length(missing)}")
            bars.append(" ".join(tokens))
        body = "|".join(bars)
        opener = "|:" if section.repeats >= 2 else "|"
        closer = ":|" if section.repeats >= 2 else "|"
        lines.append(f"{opener}{body}{closer}")
    return "\n".join(lines) + "\n"


def _abc_token(note: Note, sharps: int) -> str:
    length = _abc_length(note.duration_beats)
    if note.is_rest:
        return f"z{length}"
    return f"{_abc_pitch(note.pitch, sharps)}{length}"


def _abc_length(duration_beats) -> str:
    """Express a duration as a multiple of the L:1/8 unit (an eighth note)."""
    from fractions import Fraction

    units = Fraction(duration_beats) / Fraction(1, 2)
    if units == 1:
        return ""
    if units.denominator == 1:
        return str(units.numerator)
    if units.numerator == 1:
        return f"/{units.denominator}"
    return f"{units.numerator}/{units.denominator}"


def _abc_pitch(midi: int, sharps: int) -> str:
    """MIDI number to ABC pitch, emitting accidentals only where needed.

    ABC pitch conventions: ``C`` is middle C (C4), ``c`` is C5, ``c'`` is C6,
    ``C,`` is C3. Accidentals are relative to the key signature, so a note the
    signature already alters needs no marker, while a note that contradicts the
    signature needs an explicit one -- including a natural sign.
    """
    spelled = _spell(midi, sharps)  # e.g. "F#5" or "B-3"
    letter = spelled[0]
    accidental = spelled[1] if spelled[1] in "#-" else ""
    octave = midi // 12 - 1

    if sharps >= 0:
        altered = set(_SHARP_ORDER[:sharps])
        key_accidental = "#"
    else:
        altered = set(_FLAT_ORDER[:-sharps])
        key_accidental = "-"

    if accidental == key_accidental and letter in altered:
        marker = ""  # already implied by the key signature
    elif accidental == "" and letter in altered:
        marker = "="  # the signature would alter it; this note does not
    elif accidental == "#":
        marker = "^"
    elif accidental == "-":
        marker = "_"
    else:
        marker = ""

    if octave >= 5:
        name = letter.lower() + "'" * (octave - 5)
    else:
        name = letter + "," * (4 - octave)
    return marker + name
