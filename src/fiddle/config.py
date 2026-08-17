"""All tunable parameters in one place.

Configuration scattered across modules makes ablation studies impossible: you
cannot answer "does octave correction help?" if the octave-correction threshold
is a literal buried in a function. Every number a future experiment might want
to sweep lives here, and every stage takes its config section as an argument.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MelodyConfig:
    # "basicpitch" | "essentia" | "fiddle" | "pyin".
    #
    # Basic Pitch is the default because it measured best on real jam audio by a
    # wide margin -- it more than halves the rate at which the extracted line
    # tracks the bass, without costing repetition (see docs/RESEARCH.md). It
    # cannot share this environment, though: it pins numpy<2 and pulls
    # TensorFlow. So resolution falls back to Essentia when it is not installed,
    # loudly rather than silently -- see fiddle.melody.get_extractor.
    backend: str = "basicpitch"
    sample_rate: int = 44100
    frame_size: int = 2048
    hop_size: int = 128  # ~2.9 ms at 44.1 kHz; Melodia's documented default
    # Melodia's range gate. This is the most consequential single number in the
    # config on real recordings.
    #
    # It was originally 130 Hz (~C3), set below the fiddle deliberately so that
    # octave errors would still be visible to the cleaner. On real jam audio
    # that was a serious mistake: guitar, bass and banjo own the 100-250 Hz band
    # and are usually closer to a phone's microphone than the fiddle is, so
    # Melodia's salience function tracked *them*. The result was a contour
    # parked on sustained low drones with no melody in it at all.
    #
    # 250 Hz sits just under B3, comfortably below the fiddle's working
    # melodic range while excluding most accompaniment fundamentals. Old-time
    # fiddle melody essentially never goes below the D above the G string.
    min_frequency: float = 250.0
    max_frequency: float = 2200.0  # ~C#7, above any melody note a fiddler plays
    voicing_tolerance: float = 0.2  # Melodia default; higher = more voiced
    filter_iterations: int = 3

    # Basic Pitch's note-creation thresholds. These decide how much of the
    # model's per-frame activation survives into discrete notes, and the
    # defaults throw away a great deal: at 0.5/0.3 only about 3% of the
    # posteriorgram's energy is retained, and what survives is the loudest,
    # most sustained content -- the accompaniment (see docs/NEXT_STEPS.md).
    #
    # Lowering them does recover real material rather than noise: median note
    # duration holds near 190 ms and under 2% of notes are shorter than a
    # sixteenth. What it recovers is *simultaneous* notes -- average polyphony
    # goes 1.8 -> 6.8 -> 22 -> 36 as the pair is lowered -- so the extra notes
    # make melody selection harder, not easier, and measured worse end to end
    # with the current "loudest note above the floor" rule.
    #
    # Measured end to end against ground truth on the synthetic corpus, which
    # is the only place the question can be settled. Lowering 0.5/0.3 to
    # 0.3/0.15 improves the fundamentals and costs structure:
    #
    #                    pitch  onset   dur    key   form  octave  wrong/tune
    #   0.5 / 0.3         0.88   0.60   0.44   0.73   0.80  0.024      5.7
    #   0.3 / 0.15        0.90   0.67   0.35   0.93   0.67  0.011      2.9
    #   0.2 / 0.1         0.84   0.66   0.20   1.00   0.73  0.015      7.4
    #
    # 0.3/0.15 is the default because *which* notes and which key are right
    # matters more than how long they are: duration and form both degrade
    # because a denser pool makes the "loudest note" selector switch voices
    # more often, chopping note boundaries. Under the catalog-matching plan
    # (docs/NEXT_STEPS.md) duration and form come from the matched setting
    # anyway, while pitch and key are what the match is scored on.
    #
    # Note for anyone re-running this: the extraction-level proxies said the
    # opposite. Repetition fell from 0.29 to 0.19 on a real recording while
    # ground-truth note accuracy rose. That is the third time on this project
    # a proxy has pointed the wrong way; prefer the corpus with known answers.
    basicpitch_onset_threshold: float = 0.3
    basicpitch_frame_threshold: float = 0.15


@dataclass(frozen=True)
class CleanConfig:
    """Each stage is separately switchable so ablations are one config away."""

    enable_confidence_gate: bool = True
    min_confidence: float = 0.05  # relative to the contour's own max salience

    enable_range_gate: bool = True
    min_midi: float = 54.0  # a semitone under G3, to tolerate flat playing
    max_midi: float = 101.0

    enable_octave_correction: bool = True
    octave_window_sec: float = 0.60  # local median window for the reference pitch
    octave_tolerance_semitones: float = 1.5  # how close to exactly 12 to correct

    enable_median_filter: bool = True
    median_filter_sec: float = 0.045  # kills isolated spikes, keeps 16ths intact

    enable_vibrato_smoothing: bool = True
    # Fiddle vibrato is roughly 4-8 Hz and under ~80 cents peak-to-peak. A
    # moving average slightly longer than one vibrato cycle flattens it without
    # eating real slurs.
    vibrato_window_sec: float = 0.18
    vibrato_max_depth_cents: float = 90.0

    enable_jump_gate: bool = True
    # A fiddler can leap an octave, but not in 3 ms. Anything faster is an
    # extraction artifact.
    max_jump_semitones_per_sec: float = 600.0

    enable_short_gap_fill: bool = True
    max_gap_fill_sec: float = 0.05  # bridge dropouts inside a sustained note

    enable_drone_suppression: bool = False  # off by default; measured, not assumed
    drone_tolerance_cents: float = 45.0
    drone_min_duration_sec: float = 0.35


@dataclass(frozen=True)
class SegmentConfig:
    # A new note starts when pitch departs from the running median by more than
    # this. Half a semitone would split slides; a whole tone would merge real
    # semitone steps. 0.65 sits between them.
    pitch_change_semitones: float = 0.65
    min_note_sec: float = 0.055  # ~16ths at 170 BPM; shorter is an artifact
    min_voiced_fraction: float = 0.55
    # Rearticulated repeated notes (very common in old-time bowing) look like
    # one long note to a pitch tracker. Onsets are the only way to split them.
    use_onsets_to_split: bool = True
    onset_split_min_sec: float = 0.10
    onset_strength_percentile: float = 70.0
    merge_same_pitch_gap_sec: float = 0.03


@dataclass(frozen=True)
class RhythmConfig:
    # Old-time dance tempos. Used to fix librosa's octave errors, which are the
    # dominant tempo failure mode (it loves to report half or double).
    min_bpm: float = 90.0
    max_bpm: float = 260.0
    preferred_bpm: float = 120.0
    # Notation convention: how many contour beats per notated quarter note.
    subdivisions: tuple[int, ...] = (1, 2, 3, 4)  # quarter, 8th, triplet, 16th
    quantize_grid: int = 4  # sixteenth-note grid (4 slots per beat)
    allow_dotted: bool = True
    max_quantize_error_beats: float = 0.28


@dataclass(frozen=True)
class MeterConfig:
    candidates: tuple[str, ...] = ("2/4", "4/4")
    default: str = "4/4"
    override: str | None = None


@dataclass(frozen=True)
class KeyConfig:
    override: str | None = None
    # Old-time is strongly modal. Restricting to major/minor mislabels the huge
    # mixolydian and dorian repertoire (Old Joe Clark, June Apple, ...).
    modes: tuple[str, ...] = ("major", "mixolydian", "dorian", "minor")
    # Keys reachable in standard tuning dominate the repertoire; used as a
    # gentle prior, never as a filter.
    key_prior: dict[str, float] = field(
        default_factory=lambda: {
            "D": 1.00, "A": 0.95, "G": 0.85, "C": 0.60,
            "E": 0.40, "F": 0.30, "B-": 0.25,
        }
    )
    # Rough relative frequency of each mode in the old-time repertoire. Needed
    # because a mode and its relative minor share a pitch-class set exactly, so
    # scale fit alone cannot separate D major from B minor -- only tonic
    # evidence and this prior can.
    mode_prior: dict[str, float] = field(
        default_factory=lambda: {
            "major": 1.00, "mixolydian": 0.70, "dorian": 0.55, "minor": 0.40,
        }
    )
    prior_weight: float = 0.15


@dataclass(frozen=True)
class FormConfig:
    # Eight bars is the overwhelming norm. The others are the crooked and
    # short-section cases that genuinely occur; anything outside this list is
    # almost certainly the search carving periodic structure at the wrong scale.
    # Order is meaningful: earlier entries are treated as more plausible.
    expected_section_bars: tuple[int, ...] = (8, 4, 6, 7, 9, 12, 16)
    min_section_bars: int = 4
    # Was 20, which let the search return 15- and 16-bar "sections" on real
    # recordings. No old-time section is that long; that result was the search
    # swallowing a section and its repeat into one block.
    max_section_bars: int = 12
    # Bar lengths, in tracked beats, that a section may be notated with. The
    # repertoire is written in 2/4 and in 4/4, and the two are metrically
    # nested, so the *same audio* is correctly barred either way. Rather than
    # ask the onset envelope to settle that (it barely can -- see meter.py),
    # form detection tries both barrings of each candidate section length and
    # keeps whichever produces a plausible bar count. The 8-bar norm is a far
    # sharper discriminator than beat-three stress, so meter comes out of form
    # rather than going into it.
    meter_candidates: tuple[int, ...] = (2, 4)
    # No section of this repertoire is shorter than this many tracked beats.
    # Stated in beats rather than bars because the bar is exactly the thing
    # under dispute: 4 bars means 8 beats in 2/4 and 16 in 4/4.
    #
    # Sixteen is the shortest a real section gets under either barring -- eight
    # bars of 2/4, or four of 4/4. Twelve was tried and was a measurable
    # mistake: on a real recording it let a 14-beat block win by being read as
    # "four bars with a 2/4 bar dropped in", which scores well as a *barring*
    # while being far too short to be a section at all.
    min_section_beats: int = 16
    similarity_threshold: float = 0.62  # below this, two passes are not "the same"
    # How much better within-cluster similarity must be than between-cluster
    # before a segmentation counts as having found anything. A hypothesis under
    # this floor has separated the blocks by a margin indistinguishable from
    # noise, and emitting it is worse than emitting nothing: consensus would
    # then average unrelated music together, which is the one failure this
    # module exists to prevent.
    min_cluster_quality: float = 0.08
    # Old-time tunes are one, two or occasionally three parts. Allowing four
    # let the search escape into forms like AABBCCD that do not exist.
    max_sections: int = 3
    require_repeats: bool = False


@dataclass(frozen=True)
class ConsensusConfig:
    enabled: bool = True
    # Consensus can actively harm output if the repetitions were mis-segmented,
    # so refuse to merge passes that do not align well.
    min_alignment_score: float = 0.55
    min_observations: int = 2
    # A slot where fewer than this share of passes agree is flagged uncertain
    # rather than silently resolved.
    agreement_uncertain_below: float = 0.7
    # Weight each pass by its own extraction confidence when voting.
    weight_by_confidence: bool = True


@dataclass(frozen=True)
class ConfidenceConfig:
    # Blend weights for the final per-note confidence. Deliberately explicit so
    # calibration against the corpus can move them with evidence.
    w_salience: float = 0.30
    w_stability: float = 0.20
    w_quantization: float = 0.15
    w_agreement: float = 0.35
    uncertain_threshold: float = 0.70


@dataclass(frozen=True)
class Config:
    melody: MelodyConfig = field(default_factory=MelodyConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    meter: MeterConfig = field(default_factory=MeterConfig)
    key: KeyConfig = field(default_factory=KeyConfig)
    form: FormConfig = field(default_factory=FormConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **sections: dict[str, Any]) -> "Config":
        """Return a copy with per-section field overrides.

        >>> Config().with_overrides(consensus={"enabled": False})
        """
        updates: dict[str, Any] = {}
        for name, fields in sections.items():
            current = getattr(self, name)
            updates[name] = replace(current, **fields)
        return replace(self, **updates)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text())
        cfg = cls()
        return cfg.with_overrides(**data)


DEFAULT_CONFIG = Config()
