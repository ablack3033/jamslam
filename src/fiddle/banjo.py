"""Generate a three-finger (Scruggs-style) banjo part to accompany a fiddle tune.

This is an *arranger*, not a transcriber: it takes a canonical melody and works
out a playable five-string banjo part for it. That separation matters, because
it means the banjo feature is useful even while melody extraction is still being
improved -- point it at a hand-corrected ABC and it produces a real part.

The core idea of Scruggs style is that the right hand plays a near-continuous
stream of eighth notes in fixed patterns (rolls), and the melody emerges from
*which* notes in that stream are fretted. So the arranger works in two layers:

1. A roll pattern supplies a string for every eighth-note slot in the bar.
2. Melody notes are placed into the slots where they fall, overriding the roll's
   string choice with whatever string/fret actually produces that pitch.

Slots the melody does not claim are filled with a chord tone on the roll's
string, which is what makes the part sound like banjo rather than like a melody
with gaps. The fifth string is treated as a drone and never used for melody --
that is the instrument's defining characteristic, and using it melodically is
the most common way generated banjo parts sound wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .domain import Note, Tune

#: Open-G tuning, the standard for three-finger playing: gDGBD.
#: Indexed by string number as players count them -- 1 is the highest-pitched.
G_TUNING: dict[int, int] = {1: 62, 2: 59, 3: 55, 4: 50, 5: 67}  # D4 B3 G3 D3 g4

#: Which finger plays which string. Thumb takes the bass strings and the fifth,
#: index the second, middle the first. This is the invariant of the style.
FINGERS: dict[int, str] = {5: "T", 4: "T", 3: "T", 2: "I", 1: "M"}

#: Roll patterns, as the string played on each of the eight eighth-note slots in
#: a 4/4 bar. Simplified but idiomatic forms of the standard rolls.
ROLLS: dict[str, tuple[int, ...]] = {
    "forward": (3, 2, 1, 5, 3, 2, 1, 5),
    "backward": (1, 2, 3, 5, 1, 2, 3, 5),
    "forward_reverse": (3, 2, 5, 1, 3, 2, 5, 1),
    "alternating_thumb": (4, 2, 3, 1, 4, 2, 3, 1),
}

MAX_FRET = 12
#: Strings available for melody. The fifth is a drone and is excluded.
MELODY_STRINGS = (1, 2, 3, 4)


@dataclass(frozen=True)
class BanjoConfig:
    tuning: dict[int, int] = field(default_factory=lambda: dict(G_TUNING))
    capo: int | None = None  # None = choose automatically from the key
    roll: str = "forward"
    # Vary the roll between bars so the part does not sound mechanical.
    vary_rolls: bool = True
    # Slots per beat. Scruggs style is a stream of eighth notes.
    subdivision: int = 2


@dataclass
class BanjoNote:
    string: int
    fret: int
    pitch: int
    start_beats: Fraction
    duration_beats: Fraction
    is_melody: bool
    finger: str = "T"


@dataclass
class BanjoPart:
    notes: list[BanjoNote]
    capo: int
    key: str
    meter: str
    tuning: dict[int, int]
    measures: list[list[BanjoNote]] = field(default_factory=list)
    notes_log: list[str] = field(default_factory=list)

    @property
    def melody_coverage(self) -> float:
        """Share of the arrangement's notes that carry actual melody.

        A useful sanity number: a good Scruggs arrangement of a fiddle tune
        lands somewhere around a third to a half. Much lower means the melody
        got buried in roll filler; much higher means it is not really a roll.
        """
        if not self.notes:
            return 0.0
        return sum(1 for n in self.notes if n.is_melody) / len(self.notes)


# ---------------------------------------------------------------------------
# Capo and chord logic
# ---------------------------------------------------------------------------

_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
       "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
       "D-": 1, "E-": 3, "G-": 6, "A-": 8, "B-": 10}


def choose_capo(key: str, tuning: dict[int, int]) -> int:
    """Pick a capo position that puts the tune into open-G fingering.

    Banjo players capo rather than transpose their fingering: the open strings
    spell the I chord, and every standard lick assumes that. So the capo is
    simply the distance from G up to the tune's tonic -- 2 for A, 4 for B, 5 for
    C. D is the exception: it is a fifth above G, which would need capo 7, so
    players stay in G tuning and play D-position shapes instead.
    """
    root = key.strip().split()[0]
    tonic = _PC.get(root, 7)
    open_root = tuning[3] % 12  # the third string is the tonic in open G
    capo = (tonic - open_root) % 12
    if capo > 5:
        # Too high to be playable; play out of a lower position instead.
        return 0
    return capo


def chord_tones(key: str, melody_pitches: list[int]) -> list[int]:
    """Pitch classes of the chord that best fits this bar.

    Old-time and bluegrass backup is overwhelmingly I-IV-V, so we test only
    those three and pick whichever contains the most of the bar's melody.
    Simple, and wrong far less often than a general chord estimator would be on
    material this diatonic.
    """
    root = _PC.get(key.strip().split()[0], 7)
    candidates = {
        "I": [(root + i) % 12 for i in (0, 4, 7)],
        "IV": [(root + 5 + i) % 12 for i in (0, 4, 7)],
        "V": [(root + 7 + i) % 12 for i in (0, 4, 7)],
    }
    if not melody_pitches:
        return candidates["I"]
    best, best_hits = candidates["I"], -1
    for tones in candidates.values():
        hits = sum(1 for p in melody_pitches if p % 12 in tones)
        if hits > best_hits:
            best, best_hits = tones, hits
    return best


# ---------------------------------------------------------------------------
# Arranging
# ---------------------------------------------------------------------------


def arrange(tune: Tune, config: BanjoConfig | None = None) -> BanjoPart:
    """Arrange ``tune`` as a three-finger banjo part."""
    cfg = config or BanjoConfig()
    capo = cfg.capo if cfg.capo is not None else choose_capo(tune.key, cfg.tuning)
    open_pitches = {s: p + capo for s, p in cfg.tuning.items()}
    beats_per_bar = _beats_per_bar(tune.meter)
    slots_per_bar = int(beats_per_bar * cfg.subdivision)

    out: list[BanjoNote] = []
    measures: list[list[BanjoNote]] = []
    log: list[str] = [
        f"open-G tuning, capo {capo} (sounding key {tune.key})",
        f"roll: {cfg.roll}" + (" (varied between bars)" if cfg.vary_rolls else ""),
    ]

    # Decide one octave shift for the whole tune, not per note. A fiddle tune
    # often sits above the banjo's comfortable range, so it has to come down --
    # but folding each note independently turns stepwise motion into octave
    # leaps and destroys the melody. One shift for the whole arrangement keeps
    # the contour intact, which is what a player would actually do.
    shift = _choose_octave_shift(
        [n.pitch for s in tune.sections for n in s.notes if not n.is_rest],
        open_pitches,
    )
    if shift:
        log.append(f"melody transposed {shift:+d} semitones into banjo range")

    bar_index = 0
    for section in tune.sections:
        for measure in section.measures:
            bar_start = Fraction(bar_index * beats_per_bar)
            roll = _roll_for_bar(cfg, bar_index, slots_per_bar)
            melody = [n for n in measure.notes if not n.is_rest]
            tones = chord_tones(tune.key, [n.pitch for n in melody])
            bar_notes = _arrange_bar(
                melody, roll, slots_per_bar, bar_start, beats_per_bar,
                open_pitches, tones, cfg, shift,
            )
            measures.append(bar_notes)
            out.extend(bar_notes)
            bar_index += 1

    part = BanjoPart(notes=out, capo=capo, key=tune.key, meter=tune.meter,
                     tuning=open_pitches, measures=measures, notes_log=log)
    log.append(f"{len(out)} notes, {part.melody_coverage:.0%} of them melody")
    return part


def _arrange_bar(melody, roll, slots_per_bar, bar_start, beats_per_bar,
                 open_pitches, tones, cfg, shift: int = 0) -> list[BanjoNote]:
    slot_len = Fraction(1, cfg.subdivision)
    # Which slot each melody note claims.
    claimed: dict[int, Note] = {}
    for n in melody:
        local = n.start_beats - bar_start
        slot = int(round(float(local) * cfg.subdivision))
        if 0 <= slot < slots_per_bar and slot not in claimed:
            claimed[slot] = n

    out: list[BanjoNote] = []
    for slot in range(slots_per_bar):
        start = bar_start + slot * slot_len
        if slot in claimed:
            note = claimed[slot]
            placement = _place_pitch(note.pitch + shift, open_pitches)
            if placement is None:
                # A stray note outside the range even after the global shift.
                # Fold this one rather than dropping the melody note entirely.
                placement = _place_pitch(
                    _fold_into_range(note.pitch + shift, open_pitches), open_pitches
                )
            if placement is not None:
                string, fret = placement
                out.append(BanjoNote(
                    string=string, fret=fret,
                    pitch=open_pitches[string] + fret,
                    start_beats=start, duration_beats=slot_len,
                    is_melody=True, finger=FINGERS[string],
                ))
                continue
        # Roll filler: a chord tone on whichever string the roll wants.
        string = roll[slot % len(roll)]
        # The fifth string is a drone. Players ring it open; fretting it is
        # awkward and immediately sounds wrong.
        fret = 0 if string == 5 else _nearest_chord_fret(string, tones, open_pitches)
        out.append(BanjoNote(
            string=string, fret=fret, pitch=open_pitches[string] + fret,
            start_beats=start, duration_beats=slot_len,
            is_melody=False, finger=FINGERS[string],
        ))
    return out


def _roll_for_bar(cfg: BanjoConfig, index: int, slots: int) -> tuple[int, ...]:
    if not cfg.vary_rolls:
        return ROLLS[cfg.roll]
    # Rotate through rolls on a fixed schedule. Deterministic on purpose:
    # regenerating an arrangement must give the same part every time.
    order = [cfg.roll] + [k for k in ROLLS if k != cfg.roll]
    return ROLLS[order[index % len(order)]]


def _place_pitch(pitch: int, open_pitches: dict[int, int]) -> tuple[int, int] | None:
    """Find the easiest string/fret for a pitch, preferring low frets.

    Melody is kept off the fifth string: it is a drone that starts at the fifth
    fret and only reaches a handful of pitches. Using it melodically is the
    single most common way a generated banjo part betrays itself.
    """
    best: tuple[int, int] | None = None
    for string in MELODY_STRINGS:
        fret = pitch - open_pitches[string]
        if 0 <= fret <= MAX_FRET:
            if best is None or fret < best[1]:
                best = (string, fret)
    return best


def _choose_octave_shift(pitches: list[int], open_pitches: dict[int, int]) -> int:
    """Pick one octave transposition putting the most of the tune in low frets.

    Scored on how many notes land within the first seven frets, because that is
    where the style lives -- an arrangement that is technically in range but
    sits at the twelfth fret is not one anyone would play.
    """
    if not pitches:
        return 0
    best_shift, best_score = 0, -1
    for shift in (-24, -12, 0, 12):
        score = 0
        for p in pitches:
            placement = _place_pitch(p + shift, open_pitches)
            if placement is None:
                continue
            score += 2 if placement[1] <= 7 else 1
        if score > best_score:
            best_shift, best_score = shift, score
    return best_shift


def _fold_into_range(pitch: int, open_pitches: dict[int, int]) -> int:
    lowest = open_pitches[4]
    highest = open_pitches[1] + MAX_FRET
    while pitch < lowest:
        pitch += 12
    while pitch > highest:
        pitch -= 12
    return pitch


def _nearest_chord_fret(string: int, tones: list[int],
                        open_pitches: dict[int, int]) -> int:
    """Lowest fret on this string giving a chord tone, favouring open strings."""
    base = open_pitches[string]
    for fret in range(0, 6):
        if (base + fret) % 12 in tones:
            return fret
    return 0


def _beats_per_bar(meter: str) -> Fraction:
    num, den = meter.split("/")
    return Fraction(int(num) * 4, int(den))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def to_tablature(part: BanjoPart, bars_per_line: int = 4) -> str:
    """Render the arrangement as banjo tab.

    Tab, not notation, is what a banjo player actually reads, so this is the
    primary output for this feature. Melody notes are the ones that matter, so
    the finger row underneath makes the roll pattern visible.
    """
    lines = [
        f"Banjo arrangement -- {part.key}, {part.meter}, open-G tuning, capo {part.capo}",
        "",
    ]
    string_order = [1, 2, 3, 4, 5]
    labels = {1: "D", 2: "B", 3: "G", 4: "D", 5: "g"}

    for block_start in range(0, len(part.measures), bars_per_line):
        block = part.measures[block_start:block_start + bars_per_line]
        rows = {s: f"{labels[s]}|" for s in string_order}
        fingers = " " * 2
        for measure in block:
            for n in measure:
                for s in string_order:
                    cell = f"{n.fret:<2d}" if s == n.string else "--"
                    rows[s] += cell + "-"
                fingers += (n.finger if n.is_melody else n.finger.lower()) + "  "
            for s in string_order:
                rows[s] += "|"
            fingers += " "
        for s in string_order:
            lines.append(rows[s])
        lines.append(fingers)
        lines.append("")

    lines.append("uppercase finger = melody note, lowercase = roll filler")
    lines.append("T = thumb, I = index, M = middle")
    return "\n".join(lines)


def add_banjo_to_stream(score, part: BanjoPart):
    """Append the banjo arrangement to an existing music21 Score as a second part."""
    from music21 import clef, instrument, note as m21note, stream

    p = stream.Part()
    p.insert(0, instrument.Banjo())
    p.insert(0, clef.TrebleClef())
    beats_per_bar = _beats_per_bar(part.meter)

    for i, measure in enumerate(part.measures):
        m = stream.Measure(number=i + 1)
        for n in measure:
            el = m21note.Note(n.pitch, quarterLength=float(n.duration_beats))
            el.addLyric(n.finger if n.is_melody else n.finger.lower())
            local = n.start_beats - Fraction(i) * beats_per_bar
            m.insert(float(local), el)
        p.append(m)

    score.insert(0, p)
    return score


def export_banjo_tab(part: BanjoPart, path) -> "Path":  # noqa: F821
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_tablature(part))
    return path
