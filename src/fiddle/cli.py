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
    p_ev.add_argument("--config", type=Path, help="JSON config overrides")
    p_ev.add_argument("--json", type=Path, help="write the full report here")

    p_ab = sub.add_parser("ablate", help="compare configurations on the corpus")
    p_ab.add_argument("--difficulties", default="clean,jam,hard")
    p_ab.add_argument("--json", type=Path)

    p_bc = sub.add_parser("build-corpus", help="render the synthetic corpus to disk")
    p_bc.add_argument("out", type=Path)
    p_bc.add_argument("--difficulties", default="clean,jam,hard")

    p_id = sub.add_parser("identify", help="match a recording against a tune catalog")
    p_id.add_argument("audio", type=Path, nargs="?")
    p_id.add_argument("--from-json", type=Path,
                      help="use an existing .notes.json instead of re-transcribing")
    p_id.add_argument("--catalog", type=Path,
                      help="ABC file or directory of ABC files (default: builtin)")
    p_id.add_argument("--top", type=int, default=5)
    p_id.add_argument("--melody-backend", default=None)

    p_bj = sub.add_parser("banjo", help="arrange a three-finger banjo part")
    p_bj.add_argument("source", type=Path, help="audio file, or an .abc melody")
    p_bj.add_argument("-o", "--outdir", type=Path, default=None)
    p_bj.add_argument("--capo", type=int, default=None)
    p_bj.add_argument("--roll", default="forward",
                      choices=["forward", "backward", "forward_reverse",
                               "alternating_thumb"])
    p_bj.add_argument("--no-vary-rolls", action="store_true")

    p_bk = sub.add_parser("backends", help="list available melody backends")
    p_bk.set_defaults(func=_cmd_backends)

    # Allow `fiddle-transcribe file.m4a` with no subcommand.
    argv = list(sys.argv[1:] if argv is None else argv)
    known = {"transcribe", "eval", "ablate", "build-corpus", "backends",
             "identify", "banjo"}
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
        "identify": _cmd_identify,
        "banjo": _cmd_banjo,
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
    p.add_argument("--banjo", action="store_true",
                   help="also arrange a three-finger banjo part")
    p.add_argument("--identify", action="store_true",
                   help="also match the result against the tune catalog")
    p.add_argument("--catalog", type=Path, help="catalog for --identify")
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

    if result.presence and not result.presence.melody_found:
        from .presence import format_presence

        print(format_presence(result.presence), file=sys.stderr)
        print("", file=sys.stderr)
    print(format_summary(result))
    outputs = [".pitch.csv", ".notes.json", ".musicxml", ".mid", ".abc"]
    if args.banjo:
        from .banjo import arrange, export_banjo_tab

        part = arrange(result.tune)
        export_banjo_tab(part, f"{stem}.banjo.txt")
        outputs.append(".banjo.txt")
        print("\nbanjo: " + "; ".join(part.notes_log))
    if args.identify:
        from .catalog import load_catalog
        from .identify import format_identification, identify_result

        catalog = load_catalog(args.catalog)
        print()
        print(format_identification(identify_result(result, catalog)))
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

    # Honour --config here too. Without it the only way to measure a config
    # change against ground truth was to edit the defaults, which is exactly
    # the sort of thing that makes an ablation unreproducible.
    cfg = _config_from_args(args)
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


def _cmd_identify(args) -> int:
    """Match a transcription against a catalog of known tunes.

    Doubles as validation: hand-writing ground truth is the bottleneck in
    growing the corpus, so a confident catalog match gives us an approximate
    ground truth for free -- and a *failure* to match anything is itself a
    signal that the transcription is too noisy to be usable.
    """
    import json

    from .catalog import load_catalog
    from .identify import format_identification, identify, identify_result

    catalog = load_catalog(args.catalog)
    if not catalog:
        print(f"no tunes could be read from {args.catalog}", file=sys.stderr)
        return 2

    if args.from_json:
        payload = json.loads(args.from_json.read_text())
        pitches = [n["pitch"] for n in payload.get("raw_notes", [])]
        ident = identify(pitches, catalog, args.top)
    elif args.audio:
        from .pipeline import transcribe_file

        cfg = _config_from_args(args)
        ident = identify_result(transcribe_file(args.audio, cfg), catalog, args.top)
    else:
        print("give an audio file or --from-json", file=sys.stderr)
        return 2

    print(format_identification(ident))
    return 0


def _cmd_banjo(args) -> int:
    """Arrange a banjo part from audio or from an ABC melody.

    Accepting ABC directly matters: the arranger is useful long before melody
    extraction is reliable, because a hand-corrected ABC is a perfectly good
    input and produces a genuinely playable part.
    """
    from .banjo import BanjoConfig, arrange, export_banjo_tab, to_tablature

    if args.source.suffix.lower() == ".abc":
        tune = _tune_from_abc(args.source)
    else:
        from .pipeline import transcribe_file

        tune = transcribe_file(args.source, Config()).tune

    cfg = BanjoConfig(capo=args.capo, roll=args.roll,
                      vary_rolls=not args.no_vary_rolls)
    part = arrange(tune, cfg)
    outdir = args.outdir or args.source.parent
    export_banjo_tab(part, outdir / f"{args.source.stem}.banjo.txt")
    for line in part.notes_log:
        print(f"  {line}")
    print()
    print(to_tablature(part))
    return 0


def _tune_from_abc(path: Path):
    """Build a Tune from an ABC melody so the arranger can consume it."""
    from fractions import Fraction

    from .abc_io import beats_per_bar_from_header, meter_from_abc, parse_abc_sections
    from .domain import Measure, Note, Tune, TuneSection

    text = path.read_text()
    meter_str = meter_from_abc(text)
    beats_per_bar = float(beats_per_bar_from_header(text.splitlines()))
    sections = []
    for name, section in sorted(parse_abc_sections(text).items()):
        bars: dict[int, Measure] = {}
        for n in section.notes:
            idx = int(float(n.start_beats) // beats_per_bar)
            bars.setdefault(idx, Measure(notes=[], number=idx + 1)).notes.append(
                Note(pitch=n.pitch, start_beats=n.start_beats,
                     duration_beats=n.duration_beats, confidence=1.0)
            )
        sections.append(TuneSection(name=name,
                                    measures=[bars[k] for k in sorted(bars)],
                                    repeats=2))
    key_line = next((ln[2:].strip() for ln in text.splitlines()
                     if ln.startswith("K:")), "D")
    from .catalog import _normalize_key

    key = _normalize_key(key_line)
    from .key import _signature_for, _parse_key

    tonic, mode = _parse_key(key)
    title = next((ln[2:].strip() for ln in text.splitlines()
                  if ln.startswith("T:")), path.stem)
    return Tune(key=key, meter=meter_str, tempo=120.0, sections=sections,
                title=title, analysis={"key_sharps": _signature_for(tonic, mode)})


def _cmd_backends(args) -> int:
    from .melody import available_backends

    for name, ok in available_backends().items():
        print(f"{name:<12} {'available' if ok else 'not installed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
