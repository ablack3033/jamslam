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

No single recording clears the confidence bar (z ≥ 6).

An earlier version of this file claimed the ranking pattern was significant at
p = 0.0004. **That is retracted.** It assumed a named tune was the correct
answer for every recording, and `memory_of_home` is probably *not* Old Joe
Clark — so that row is a false positive and the premise fails.

Measured chance rates over random input: the best of these three tunes lands in
the top two 31% of the time by luck alone; Old Joe Clark reaches rank 1 on 5.3%
of random inputs and Sandy River Belle on 3.7%. That leaves one plausible hit
(`bebop_1`, p ≈ 0.04) and one likely miss (`memory_of_home`, p ≈ 0.05) at
comparable significance — **no net evidence either way.**

What is still open:

1. **The mapping is unconfirmed.** Three names for five recordings means at
   least two share a tune. `bebop_1` may be Sandy River Belle; `memory_of_home`
   is probably not Old Joe Clark. The rest is unknown.
2. **Cluck Old Hen ranks nowhere** on any recording, and it is also the *least*
   likely of the three to match by chance (6% top-2 rate against 10% uniform),
   so its absence is weak evidence rather than none.
3. **Verified ground truth for one recording** would settle far more than
   further statistics can.
