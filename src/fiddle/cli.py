"""Command-line interface: the research tool, and the first milestone.

    fiddle-transcribe recording.m4a

produces recording.pitch.csv, recording.notes.json, recording.musicxml,
recording.mid and (with --plots) a diagnostic figure.

The other subcommands exist so that measurement is as easy as transcription --
``eval`` and ``ablate`` are the commands that decide whether an algorithm change
was an improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fiddle-transcribe",
        description="Infer the canonical fiddle melody from an old-time jam recording.",
    )
    sub = parser.add_subparsers(dest="command")

    p_tr = sub.add_parser("transcribe", help="transcribe an audio file (default)")
    _add_transcribe_args(p_tr)

    p_ev = sub.add_parser("eval", help="score the pipeline against ground truth")
    p_ev.add_argument("--dataset", type=Path,
                      help="directory of real recordings with expected.abc files")
    p_ev.add_argument("--difficulties", default="clean,jam,hard")
    p_ev.add_argument("--tune", action="append", help="restrict to these tune slugs")
    p_ev.add_argument("--backend", default=None, help="melody backend override")
    p_ev.add_argument("--json", type=Path, help="write the full report here")

    p_ab = sub.add_parser("ablate", help="compare configurations on the corpus")
    p_ab.add_argument("--difficulties", default="clean,jam,hard")
    p_ab.add_argument("--json", type=Path)

    p_bc = sub.add_parser("build-corpus", help="render the synthetic corpus to disk")
    p_bc.add_argument("out", type=Path)
    p_bc.add_argument("--difficulties", default="clean,jam,hard")

    p_bk = sub.add_parser("backends", help="list available melody backends")
    p_bk.set_defaults(func=_cmd_backends)

    # Allow `fiddle-transcribe file.m4a` with no subcommand.
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"transcribe", "eval", "ablate", "build-corpus", "backends"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv.insert(0, "transcribe")
    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)
    handlers = {
        "transcribe": _cmd_transcribe,
        "eval": _cmd_eval,
        "ablate": _cmd_ablate,
        "build-corpus": _cmd_build_corpus,
        "backends": _cmd_backends,
    }
    return handlers[args.command](args)


def _add_transcribe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", type=Path)
    p.add_argument("-o", "--outdir", type=Path, default=None,
                   help="output directory (default: alongside the input)")
    p.add_argument("--melody-backend", default=None,
                   choices=["essentia", "pyin", "basicpitch"])
    p.add_argument("--key", default=None, help="override the inferred key")
    p.add_argument("--meter", default=None, choices=["2/4", "4/4"])
    p.add_argument("--no-consensus", action="store_true",
                   help="skip consensus and use the single best pass")
    p.add_argument("--plots", action="store_true", help="write a diagnostic figure")
    p.add_argument("--config", type=Path, help="JSON config overrides")


def _config_from_args(args) -> Config:
    cfg = Config.from_json(args.config) if getattr(args, "config", None) else Config()
    overrides: dict = {}
    if getattr(args, "melody_backend", None):
        overrides["melody"] = {"backend": args.melody_backend}
    if getattr(args, "key", None):
        overrides["key"] = {"override": args.key}
    if getattr(args, "meter", None):
        overrides["meter"] = {"override": args.meter}
    if getattr(args, "no_consensus", False):
        overrides["consensus"] = {"enabled": False}
    return cfg.with_overrides(**overrides) if overrides else cfg


def _cmd_transcribe(args) -> int:
    from .diagnostics import (
        format_summary,
        plot_diagnostics,
        write_notes_json,
        write_pitch_csv,
    )
    from .pipeline import transcribe_file
    from .score import export_abc, export_midi, export_musicxml

    if not args.audio.exists():
        print(f"no such file: {args.audio}", file=sys.stderr)
        return 2

    cfg = _config_from_args(args)
    result = transcribe_file(args.audio, cfg)
    # The filename is the only tune name we have; a jam recording is usually
    # named after the tune, and an approximate title beats "Untitled" on a score.
    result.tune.title = args.audio.stem.replace("_", " ").replace("-", " ").title()

    outdir = args.outdir or args.audio.parent
    stem = outdir / args.audio.stem
    write_pitch_csv(result, f"{stem}.pitch.csv")
    write_notes_json(result, f"{stem}.notes.json")
    export_musicxml(result.tune, f"{stem}.musicxml")
    export_midi(result.tune, f"{stem}.mid")
    export_abc(result.tune, f"{stem}.abc")

    print(format_summary(result))
    outputs = [".pitch.csv", ".notes.json", ".musicxml", ".mid", ".abc"]
    if args.plots:
        if plot_diagnostics(result, f"{stem}.diagnostics.png"):
            outputs.append(".diagnostics.png")
        else:
            print("\n(matplotlib not installed; skipping plots)", file=sys.stderr)
    print("\nwrote: " + ", ".join(f"{args.audio.stem}{s}" for s in outputs))
    return 0


def _cmd_eval(args) -> int:
    from .corpus.library import ALL_TUNES, get_tune
    from .eval.runner import format_table, run_dataset, run_synthetic, save_report

    cfg = Config()
    if args.backend:
        cfg = cfg.with_overrides(melody={"backend": args.backend})

    if args.dataset:
        result = run_dataset(args.dataset, cfg)
        if not result.scores:
            print(f"no scorable tunes found in {args.dataset}.\n"
                  f"Each subdirectory needs a recording.* and an expected.abc.",
                  file=sys.stderr)
            return 2
    else:
        tunes = [get_tune(s) for s in args.tune] if args.tune else list(ALL_TUNES)
        result = run_synthetic(
            cfg, tunes=tunes,
            difficulties=tuple(args.difficulties.split(",")),
        )
    print(format_table(result))
    if args.json:
        save_report(result, args.json)
        print(f"\nwrote {args.json}")
    return 0


def _cmd_ablate(args) -> int:
    """Compare configurations that isolate one design decision each.

    This is the experiment that tests the project's central hypothesis: that
    domain priors plus consensus across repeats beat a generic approach.
    """
    from .eval.runner import compare, run_synthetic

    difficulties = tuple(args.difficulties.split(","))
    base = Config()
    variants = {
        "full": base,
        "no-consensus": base.with_overrides(consensus={"enabled": False}),
        "no-cleaning": base.with_overrides(clean={
            "enable_octave_correction": False,
            "enable_median_filter": False,
            "enable_vibrato_smoothing": False,
            "enable_jump_gate": False,
            "enable_short_gap_fill": False,
        }),
        "no-onset-split": base.with_overrides(segment={"use_onsets_to_split": False}),
        "pyin": base.with_overrides(melody={"backend": "pyin"}),
    }

    results = {}
    for name, cfg in variants.items():
        print(f"running {name}...", file=sys.stderr)
        results[name] = run_synthetic(cfg, difficulties=difficulties, name=name)
    print(compare(results))
    if args.json:
        args.json.write_text(json.dumps(
            {n: r.aggregate() for n, r in results.items()}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def _cmd_build_corpus(args) -> int:
    """Render the synthetic corpus to disk in the datasets/ layout."""
    from .audio import write_wav
    from .corpus.library import ALL_TUNES
    from .corpus.synth import render_tune

    for tune in ALL_TUNES:
        for difficulty in args.difficulties.split(","):
            d = args.out / f"{tune.slug}__{difficulty}"
            d.mkdir(parents=True, exist_ok=True)
            audio, truth = render_tune(tune, difficulty=difficulty, seed=1)
            write_wav(d / "recording.wav", audio)
            (d / "expected.abc").write_text(tune.abc)
            (d / "meta.json").write_text(json.dumps({
                "key": truth.key, "meter": truth.meter,
                "tempo_bpm": truth.tempo_bpm, "form": truth.form,
                "difficulty": difficulty, "synthetic": True,
            }, indent=2))
            (d / "truth.json").write_text(json.dumps(truth.to_dict(), indent=2))
            print(d)
    return 0


def _cmd_backends(args) -> int:
    from .melody import available_backends

    for name, ok in available_backends().items():
        print(f"{name:<12} {'available' if ok else 'not installed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
