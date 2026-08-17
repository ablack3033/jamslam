"""Transcription metrics, reported separately and never collapsed into one number.

A single "accuracy" figure hides exactly the information needed to improve the
system. A transcription that gets every pitch right but is a beat out of phase
is a rhythm bug; one with the right rhythm and octave-displaced pitches is an
extraction bug. Those need different fixes, so they get different numbers.

What is measured here:

  pitch          fraction of aligned notes with the exact right MIDI pitch
  pitch_class    same, ignoring octave -- the gap between the two IS the
                 octave-error rate, which is our biggest suspected failure mode
  onset          fraction of truth notes matched within a beat tolerance
  duration       fraction of matched notes with the right notated duration
  sequence       1 - normalized edit distance over the melodic sequence
  form           whether section count, length and repeat structure were right
  key/meter/tempo exact-match and relative-error checks
  calibration    does low confidence actually predict wrong notes?

The last one is the product-critical metric. The stated value proposition is
"correct 2-5 flagged notes instead of transcribing the whole tune", which is
only true if the flags are where the errors are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..abc_io import TruthSection
from ..domain import Note, Tune

# A note is considered to be at the right place if it starts within this many
# beats of the truth. An eighth note is 0.5 beats, so 0.3 is tight enough to
# catch a genuine rhythmic error but tolerant of one grid slot of jitter.
ONSET_TOLERANCE_BEATS = 0.3


@dataclass
class SectionScore:
    truth_label: str
    predicted_label: str
    n_truth: int
    n_predicted: int
    n_matched: int
    pitch_accuracy: float
    pitch_class_accuracy: float
    onset_accuracy: float
    duration_accuracy: float
    sequence_similarity: float
    insertions: int
    deletions: int


@dataclass
class CalibrationScore:
    """Does the confidence signal actually rank errors to the top?"""

    n_notes: int
    mean_confidence_correct: float
    mean_confidence_wrong: float
    # Of the notes we flagged uncertain, how many were genuinely wrong.
    flag_precision: float
    # Of the genuinely wrong notes, how many we flagged.
    flag_recall: float
    n_flagged: int
    n_wrong: int
    # Probability that a random wrong note scores below a random right one.
    # 0.5 is useless, 1.0 is perfect ranking. This is the headline number for
    # the "we know where we are unsure" claim.
    auc: float


@dataclass
class TranscriptionScore:
    slug: str
    difficulty: str
    pitch_accuracy: float
    pitch_class_accuracy: float
    onset_accuracy: float
    duration_accuracy: float
    sequence_similarity: float
    octave_error_rate: float
    key_correct: bool
    key_predicted: str
    key_truth: str
    meter_correct: bool
    tempo_error_pct: float
    form_correct: bool
    form_predicted: str
    form_truth: str
    n_sections_predicted: int
    n_sections_truth: int
    calibration: CalibrationScore | None = None
    sections: list[SectionScore] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def score_transcription(
    tune: Tune,
    truth_sections: dict[str, TruthSection],
    *,
    truth_key: str,
    truth_meter: str,
    truth_tempo: float,
    truth_form: str,
    slug: str = "",
    difficulty: str = "",
    uncertain_threshold: float = 0.7,
) -> TranscriptionScore:
    """Compare a predicted Tune against canonical ground-truth sections."""
    pairs = _match_sections(tune, truth_sections)

    section_scores: list[SectionScore] = []
    all_matches: list[tuple[Note, int]] = []  # (predicted note, truth pitch)
    for pred_section, truth in pairs:
        pred_notes = [n for n in pred_section.notes if not n.is_rest] if pred_section else []
        truth_notes = list(truth.notes)
        alignment = align_sequences(
            [(n.pitch, float(n.start_beats)) for n in pred_notes],
            [(n.pitch, float(n.start_beats)) for n in truth_notes],
        )
        section_scores.append(
            _score_section(pred_section, truth, pred_notes, truth_notes, alignment)
        )
        for pi, ti in alignment:
            if pi is not None and ti is not None:
                all_matches.append((pred_notes[pi], truth_notes[ti].pitch))

    def agg(attr: str) -> float:
        vals = [getattr(s, attr) for s in section_scores]
        weights = [max(1, s.n_truth) for s in section_scores]
        return float(np.average(vals, weights=weights)) if vals else 0.0

    pitch_acc = agg("pitch_accuracy")
    pc_acc = agg("pitch_class_accuracy")
    # The gap between pitch-class and pitch accuracy is exactly the share of
    # notes we got right modulo an octave, i.e. the octave-error rate.
    octave_rate = max(0.0, pc_acc - pitch_acc)

    return TranscriptionScore(
        slug=slug,
        difficulty=difficulty,
        pitch_accuracy=pitch_acc,
        pitch_class_accuracy=pc_acc,
        onset_accuracy=agg("onset_accuracy"),
        duration_accuracy=agg("duration_accuracy"),
        sequence_similarity=agg("sequence_similarity"),
        octave_error_rate=octave_rate,
        key_correct=_keys_equal(tune.key, truth_key),
        key_predicted=tune.key,
        key_truth=truth_key,
        meter_correct=tune.meter == truth_meter,
        tempo_error_pct=abs(tune.tempo - truth_tempo) / max(truth_tempo, 1e-6) * 100.0,
        form_correct=_form_equivalent(tune.form, truth_form),
        form_predicted=tune.form,
        form_truth=truth_form,
        n_sections_predicted=len(tune.sections),
        n_sections_truth=len(truth_sections),
        calibration=_calibration(all_matches, uncertain_threshold),
        sections=section_scores,
    )


def _score_section(pred_section, truth, pred_notes, truth_notes, alignment) -> SectionScore:
    matched = [(pi, ti) for pi, ti in alignment if pi is not None and ti is not None]
    n_truth = len(truth_notes)

    correct_pitch = sum(1 for pi, ti in matched
                        if pred_notes[pi].pitch == truth_notes[ti].pitch)
    correct_pc = sum(1 for pi, ti in matched
                     if pred_notes[pi].pitch % 12 == truth_notes[ti].pitch % 12)
    correct_onset = sum(
        1 for pi, ti in matched
        if abs(float(pred_notes[pi].start_beats) - float(truth_notes[ti].start_beats))
        <= ONSET_TOLERANCE_BEATS
    )
    correct_dur = sum(
        1 for pi, ti in matched
        if pred_notes[pi].duration_beats == truth_notes[ti].duration_beats
    )

    denom = max(1, n_truth)
    return SectionScore(
        truth_label=truth.name,
        predicted_label=pred_section.name if pred_section else "-",
        n_truth=n_truth,
        n_predicted=len(pred_notes),
        n_matched=len(matched),
        pitch_accuracy=correct_pitch / denom,
        pitch_class_accuracy=correct_pc / denom,
        onset_accuracy=correct_onset / denom,
        duration_accuracy=correct_dur / denom,
        sequence_similarity=_sequence_similarity(
            [n.pitch for n in pred_notes], [n.pitch for n in truth_notes]
        ),
        insertions=sum(1 for pi, ti in alignment if ti is None),
        deletions=sum(1 for pi, ti in alignment if pi is None),
    )


def align_sequences(
    pred: list[tuple[int, float]], truth: list[tuple[int, float]]
) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch alignment of two note sequences.

    Global alignment rather than nearest-neighbour matching, because a single
    inserted or dropped note early in a section would otherwise misalign
    everything after it and report near-zero accuracy for a nearly correct
    transcription. Scoring combines pitch identity with beat proximity so that
    the same pitch in the wrong place is not treated as a free match.
    """
    n, m = len(pred), len(truth)
    if n == 0 or m == 0:
        return [(i, None) for i in range(n)] + [(None, j) for j in range(m)]

    gap = -1.0
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    ptr[:, 0] = 1
    ptr[0, :] = 2

    for i in range(1, n + 1):
        pp, pt = pred[i - 1]
        for j in range(1, m + 1):
            tp, tt = truth[j - 1]
            if pp == tp:
                s = 2.0
            elif pp % 12 == tp % 12:
                s = 0.5  # octave error: related, but wrong
            else:
                s = -1.0
            # Penalize matching notes that are far apart in time.
            s -= min(2.0, abs(pt - tt) * 0.5)
            diag = score[i - 1, j - 1] + s
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            ptr[i, j] = 0 if best == diag else (1 if best == up else 2)

    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i, j] == 0:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and ptr[i, j] == 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return list(reversed(out))


def _sequence_similarity(a: list[int], b: list[int]) -> float:
    """1 - normalized Levenshtein distance over pitch sequences."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def _match_sections(tune: Tune, truth_sections: dict[str, TruthSection]):
    """Pair predicted sections with truth sections by content, not by name.

    Section letters are arbitrary labels the form detector assigned in order of
    appearance; scoring must not punish a correct transcription for calling the
    same music "A" when the ground truth called it "B".
    """
    truth_list = list(truth_sections.values())
    pred_list = list(tune.sections)
    pairs = []
    used: set[int] = set()
    for truth in truth_list:
        best_i, best_s = None, -1.0
        for i, pred in enumerate(pred_list):
            if i in used:
                continue
            s = _sequence_similarity(
                [n.pitch for n in pred.notes if not n.is_rest],
                [n.pitch for n in truth.notes],
            )
            if s > best_s:
                best_i, best_s = i, s
        if best_i is not None:
            used.add(best_i)
            pairs.append((pred_list[best_i], truth))
        else:
            pairs.append((None, truth))
    return pairs


def _calibration(
    matches: list[tuple[Note, int]], threshold: float
) -> CalibrationScore | None:
    if not matches:
        return None
    conf = np.array([m[0].confidence for m in matches])
    wrong = np.array([m[0].pitch != m[1] for m in matches])

    flagged = conf < threshold
    n_flagged = int(flagged.sum())
    n_wrong = int(wrong.sum())
    flag_precision = float(wrong[flagged].mean()) if n_flagged else 0.0
    flag_recall = float(flagged[wrong].mean()) if n_wrong else 0.0

    return CalibrationScore(
        n_notes=len(matches),
        mean_confidence_correct=float(conf[~wrong].mean()) if (~wrong).any() else 0.0,
        mean_confidence_wrong=float(conf[wrong].mean()) if wrong.any() else 0.0,
        flag_precision=flag_precision,
        flag_recall=flag_recall,
        n_flagged=n_flagged,
        n_wrong=n_wrong,
        auc=_auc(conf, wrong),
    )


def _auc(confidence: np.ndarray, wrong: np.ndarray) -> float:
    """P(confidence of a wrong note < confidence of a correct note).

    Computed directly from all pairs via rank statistics; ties count as half,
    which matters because confidence values repeat often.
    """
    pos = confidence[~wrong]  # correct notes
    neg = confidence[wrong]  # wrong notes
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks within ties so ties contribute 0.5 rather than 1 or 0.
    values = np.concatenate([pos, neg])
    for v in np.unique(values):
        mask = values == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _keys_equal(a: str, b: str) -> bool:
    """Compare keys, tolerating enharmonic spelling but not mode differences."""
    def norm(k: str) -> tuple[str, str]:
        parts = k.strip().split()
        root = parts[0].replace("b", "-")
        mode = parts[1].lower() if len(parts) > 1 else "major"
        enh = {"D-": "C#", "E-": "D#", "G-": "F#", "A-": "G#", "B-": "A#"}
        return enh.get(root, root), mode
    return norm(a) == norm(b)


def _form_equivalent(predicted: str, truth: str) -> bool:
    """Compare form strings up to how many times the jam repeated the tune.

    "AABB" and "AABBAABB" describe the same tune; the second just went round
    twice. What must match is the *pattern*, so both are reduced to their
    shortest repeating unit before comparison.
    """
    return _reduce_form(predicted) == _reduce_form(truth)


def _reduce_form(form: str) -> str:
    if not form:
        return ""
    for size in range(1, len(form) + 1):
        if len(form) % size == 0 and form[:size] * (len(form) // size) == form:
            return form[:size]
    return form
