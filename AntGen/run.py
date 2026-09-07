#!/usr/bin/env python
"""AntGen command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ANTGEN_ROOT = Path(__file__).resolve().parent
WORKSPACE = ANTGEN_ROOT.parent
for path in (str(ANTGEN_ROOT), str(WORKSPACE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from antgen.metrics import analyze_fullwave_results
from antgen.pipeline import generate_candidates, make_run_dir, run_matlab_fullwave
from antgen.visualization import visualize_fullwave_stage
from config import CONFIG


def _config(section: str, key: str):
    try:
        return CONFIG[section][key]
    except KeyError as exc:
        raise KeyError(f"AntGen/config.py is missing CONFIG[{section!r}][{key!r}]") from exc


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "S11/pattern -> diffusion current -> topology/feed -> CNN ranking -> MATLAB validation. "
            "Defaults come from AntGen/config.py."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "infer", "simulate", "analyze"),
        default=_config("run", "stage"),
        help="Run the entire pipeline or one resumable stage",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_config("run", "run_dir"),
        help="Existing run directory for simulate/analyze, or an explicit new directory for all/infer",
    )
    parser.add_argument("--output-root", type=Path, default=_config("run", "output_root"))
    parser.add_argument("-m", "--num-conditions", type=int, default=_config("sampling", "num_conditions"))
    parser.add_argument("-n", "--num-candidates", type=int, default=_config("sampling", "num_candidates"))
    parser.add_argument("--test-start", type=int, default=_config("sampling", "test_start"))
    parser.add_argument(
        "--selection",
        choices=("sequential", "random"),
        default=_config("sampling", "selection"),
    )
    parser.add_argument("--seed", type=int, default=_config("sampling", "seed"))
    parser.add_argument(
        "--device",
        default=_config("inference", "device"),
        help="auto, cpu, cuda, or cuda:<id>",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=_config("inference", "generation_batch_size"),
    )
    parser.add_argument(
        "--diffusion-progress",
        action=argparse.BooleanOptionalAction,
        default=_config("inference", "diffusion_progress"),
    )
    parser.add_argument(
        "--s11-cfg-scale", type=float, default=_config("inference", "s11_cfg_scale")
    )
    parser.add_argument(
        "--pattern-cfg-scale", type=float, default=_config("inference", "pattern_cfg_scale")
    )
    parser.add_argument(
        "--metal-threshold", type=float, default=_config("inference", "metal_threshold")
    )
    parser.add_argument(
        "--topology-postprocess",
        choices=("feed-component", "none"),
        default=_config("inference", "topology_postprocess"),
        help="Optionally retain only the four-connected metal component containing the feed",
    )
    parser.add_argument("--h5-path", type=Path, default=_config("paths", "h5_path"))
    parser.add_argument(
        "--diffusion-checkpoint", type=Path, default=_config("paths", "diffusion_checkpoint")
    )
    parser.add_argument(
        "--structure-checkpoint", type=Path, default=_config("paths", "structure_checkpoint")
    )
    parser.add_argument(
        "--cnn-surrogate-checkpoint",
        type=Path,
        default=_config("paths", "cnn_surrogate_checkpoint"),
    )
    parser.add_argument(
        "--surrogate-batch-size", type=int, default=_config("surrogate", "batch_size")
    )
    parser.add_argument("--matlab-command", default=_config("matlab", "command"))
    parser.add_argument(
        "--matlab-workers",
        type=int,
        default=_config("matlab", "workers"),
        help="0=MATLAB default parallel pool, 1=serial, >1=exact pool size",
    )
    parser.add_argument(
        "--s11-metric-weight", type=float, default=_config("metrics", "s11_weight")
    )
    parser.add_argument(
        "--pattern-metric-weight", type=float, default=_config("metrics", "pattern_weight")
    )
    parser.add_argument(
        "--visualization-enabled",
        action=argparse.BooleanOptionalAction,
        default=_config("visualization", "enabled"),
    )
    parser.add_argument(
        "--max-visualized-conditions",
        type=int,
        default=_config("visualization", "max_conditions"),
    )
    parser.add_argument(
        "--candidate-page-size",
        type=int,
        default=_config("visualization", "candidate_page_size"),
    )
    parser.add_argument(
        "--pattern-frequency-index",
        type=int,
        default=_config("visualization", "pattern_frequency_index"),
    )
    parser.add_argument("--figure-dpi", type=int, default=_config("visualization", "dpi"))
    return parser.parse_args(argv)


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.stage in {"simulate", "analyze"}:
        if args.run_dir is None:
            raise ValueError(f"--run-dir is required for --stage {args.stage}")
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        return run_dir
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"New inference run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return make_run_dir(args.output_root.resolve())


def _validate_args(args: argparse.Namespace) -> None:
    if args.stage not in {"all", "infer", "simulate", "analyze"}:
        raise ValueError("CONFIG['run']['stage'] must be all, infer, simulate, or analyze")
    if args.num_conditions <= 0:
        raise ValueError("CONFIG sampling.num_conditions (m) must be > 0")
    if args.num_candidates <= 0:
        raise ValueError("CONFIG sampling.num_candidates (n) must be > 0")
    if args.test_start < 0:
        raise ValueError("CONFIG sampling.test_start must be >= 0")
    if args.selection not in {"sequential", "random"}:
        raise ValueError("CONFIG sampling.selection must be sequential or random")
    if args.generation_batch_size <= 0:
        raise ValueError("CONFIG inference.generation_batch_size must be > 0")
    if args.surrogate_batch_size <= 0:
        raise ValueError("CONFIG surrogate.batch_size must be > 0")
    if not 0.0 <= args.metal_threshold <= 1.0:
        raise ValueError("CONFIG inference.metal_threshold must be in [0,1]")
    if args.topology_postprocess not in {"feed-component", "none"}:
        raise ValueError("CONFIG inference.topology_postprocess must be feed-component or none")
    if args.matlab_workers < 0:
        raise ValueError("CONFIG matlab.workers must be >= 0")
    if (
        args.s11_metric_weight < 0
        or args.pattern_metric_weight < 0
        or args.s11_metric_weight + args.pattern_metric_weight <= 0
    ):
        raise ValueError("CONFIG metric weights must be non-negative and not both zero")
    if args.max_visualized_conditions <= 0:
        raise ValueError("CONFIG visualization.max_conditions must be > 0")
    if args.candidate_page_size <= 0:
        raise ValueError("CONFIG visualization.candidate_page_size must be > 0")
    if args.pattern_frequency_index < 0:
        raise ValueError("CONFIG visualization.pattern_frequency_index must be >= 0")
    if args.figure_dpi <= 0:
        raise ValueError("CONFIG visualization.dpi must be > 0")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_dir = _resolve_run_dir(args)
    print(f"[AntGen] run directory: {run_dir}")

    if args.stage in {"all", "infer"}:
        generate_candidates(args, WORKSPACE, run_dir)

    if args.stage in {"all", "simulate"}:
        run_matlab_fullwave(
            ANTGEN_ROOT,
            run_dir,
            matlab_command=args.matlab_command,
            matlab_workers=args.matlab_workers,
        )

    if args.stage in {"all", "simulate", "analyze"}:
        summary = analyze_fullwave_results(
            run_dir,
            s11_weight=args.s11_metric_weight,
            pattern_weight=args.pattern_metric_weight,
        )
        if args.visualization_enabled:
            visualize_fullwave_stage(
                run_dir,
                s11_weight=args.s11_metric_weight,
                pattern_weight=args.pattern_metric_weight,
                max_conditions=args.max_visualized_conditions,
                pattern_frequency_index=args.pattern_frequency_index,
                dpi=args.figure_dpi,
            )
        print(
            json.dumps(
                summary["fullwave_validation"]["mean_fullwave_metrics_of_selected"],
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.stage == "infer":
        print(
            "[AntGen] inference complete. To continue, set CONFIG run.stage='simulate' and "
            f"run.run_dir=r'{run_dir}', then run.py again. CLI equivalent: "
            f'python "{Path(__file__).resolve()}" --stage simulate --run-dir "{run_dir}"'
        )


if __name__ == "__main__":
    main()
