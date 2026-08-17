# Candidate ground-truth settings

The three tunes named as being present in the real recordings:

    old_joe_clark.abc
    cluck_old_hen.abc
    sandy_river_belle.abc

**These are approximations typed from memory, not verified transcriptions of
what is actually played on the recordings.** They are good enough to identify a
tune, which is a coarse judgement, and *not* good enough to score a
transcription against, which is a fine one. Two specific warnings:

- `sandy_river_belle.abc` is a **6-bar crooked setting that was invented** to
  exercise crooked-tune handling in the test corpus. It is plausible but it is
  not a documented setting. Treat it as fiction until checked.
- All three encode one version of a tune that has many. Old-time settings differ
  between players in ornaments, passing notes, and sometimes whole phrases.

Before using any of these as `expected.abc`, check it against the recording by
ear and correct it. The pipeline's own `.abc` export is designed to make that
edit cheap.

## What the recordings say

Identification ranks, against the full 20-tune catalog. `z` is standard
deviations above what that tune scores on meaningless input:

| recording | best named tune | z | rank of 20 |
|---|---|---|---|
| bebop_1 | Sandy River Belle | 1.5 | **1** |
| bebop_2 | Sandy River Belle | −0.4 | 2 |
| bebop_4 | Old Joe Clark | 1.8 | 2 |
| bebop_5 | Old Joe Clark | 1.8 | 2 |
| memory_of_home | Old Joe Clark | 2.7 | **1** |

No single recording clears the confidence bar (z ≥ 6). But a named tune lands in
the top two of twenty for **every** recording, which a permutation test puts at
**p = 0.0004**. So the transcriptions do carry real melodic information — just
not enough to identify any one recording on its own.

Two things this leaves open:

1. **The mapping is unconfirmed.** Three tune names were given for five
   recordings, so at least two recordings share a tune, and which is which is
   inferred from the table above rather than known.
2. **Cluck Old Hen ranks nowhere**, on any recording. Either none of these five
   is that tune, or our setting for it is poor, or it is one of the recordings
   we are ranking wrongly.
