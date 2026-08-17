"""Does the fiddle survive in Basic Pitch's RAW output, before thresholding?

Basic Pitch produces per-frame posteriorgrams (note / onset / contour) and only
then thresholds them into discrete note events. The previous measurement showed
the *events* are 86% two pitch classes. If the posteriorgram is materially
richer, then the information is being destroyed at the thresholding step and
catalog matching has a real surface to match against. If it is equally
collapsed, the fiddle is not recoverable from these recordings at all.

Answer, on real jam audio: the posteriorgram is far richer (3.55 bits against
1.58 after thresholding), and the default threshold discards 97% of the energy.
Keep this script -- it is the measurement any change to the front end should be
checked against. See docs/NEXT_STEPS.md.

Usage:  python tools/measure_posteriorgram.py memory_of_home
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from fiddle.audio import load_audio, preprocess
from fiddle.config import Config

cfg = Config()
name = sys.argv[1]
audio = preprocess(load_audio(Path("/home/user/jamslam/work/real") / f"{name}.m4a",
                              cfg.melody.sample_rate))

from basic_pitch.inference import predict

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, audio.samples, audio.sample_rate)
    model_output, _, events = predict(f.name)

note = np.asarray(model_output["note"])       # (frames, 88) note activations
print(f"{name}: posteriorgram {note.shape}, {len(events)} thresholded events")

MIDI_MIN = 21  # Basic Pitch's lowest bin is A0


def pc_stats(weight: np.ndarray, label: str) -> None:
    """Pitch-class entropy of an energy distribution over the 88 bins."""
    pcs = np.zeros(12)
    for i, w in enumerate(weight):
        pcs[(MIDI_MIN + i) % 12] += w
    total = pcs.sum()
    if total <= 0:
        print(f"  {label:38s} (empty)")
        return
    p = pcs / total
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    top2 = float(np.sort(p)[-2:].sum())
    print(f"  {label:38s} entropy {ent:4.2f} bits | top-2 pc {top2:5.1%}")


# Total activation per pitch bin, over the whole recording.
pc_stats(note.sum(axis=0), "raw posteriorgram (all energy)")

# Restricted to the melodic register (D4 = 62 -> bin 41).
floor_bin = 64 - MIDI_MIN
masked = note.copy()
masked[:, :floor_bin] = 0.0
pc_stats(masked.sum(axis=0), "posteriorgram, melodic register")

# The thresholding step, approximated at several confidence levels: this is
# where the pipeline currently commits.
for thresh in (0.3, 0.5, 0.7):
    gated = np.where(note >= thresh, note, 0.0)
    pc_stats(gated.sum(axis=0), f"thresholded at {thresh}")

# How much energy sits BELOW the default threshold -- i.e. is being discarded.
for thresh in (0.3, 0.5):
    kept = note[note >= thresh].sum()
    print(f"  energy retained at threshold {thresh}: {kept / note.sum():.1%}")

# Per-frame: how often is the strongest melodic-register bin something other
# than the two dominant pitch classes? That is the fiddle's chance of survival.
pcs = np.zeros(12)
for i in range(floor_bin, note.shape[1]):
    pcs[(MIDI_MIN + i) % 12] += note[:, i].sum()
dominant = set(np.argsort(pcs)[-2:])
peak_bins = masked.argmax(axis=1)
peak_pc = (MIDI_MIN + peak_bins) % 12
voiced = masked.max(axis=1) > 0.3
other = np.mean([pc not in dominant for pc, v in zip(peak_pc, voiced) if v]) \
    if voiced.any() else 0.0
print(f"  frames whose peak is NOT a dominant pitch class: {other:.1%} "
      f"({voiced.sum()} voiced frames)")
