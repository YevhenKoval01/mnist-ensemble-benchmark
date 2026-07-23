"""Command-line interface for the MNIST ensemble benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mnist_benchmark.data import load_mnist_csv, split_dataset
from mnist_benchmark.experiment import (
    MODEL_NAMES,
    run_model_selection,
    save_artifacts,
    save_dataset_plots,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "mnist_sample.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark tree-based classifiers on an MNIST CSV dataset.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Refit tuned models using one-pixel image translations.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots in addition to saving them.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )

    try:
        features, labels, feature_names = load_mnist_csv(
            args.data,
            sample_limit=args.sample_limit,
            random_state=args.random_state,
        )
        split = split_dataset(
            features,
            labels,
            feature_names,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        save_dataset_plots(split, args.output_dir, show_plots=args.show_plots)
        artifacts = run_model_selection(
            split,
            selected_models=tuple(args.models),
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            augment=args.augment,
        )
        save_artifacts(artifacts, split, args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        return 1

    best_result = next(
        result
        for result in artifacts.results
        if result.name == artifacts.selected_model_name
    )
    final_accuracy = (
        best_result.augmented_test_accuracy
        if args.augment and best_result.augmented_test_accuracy is not None
        else best_result.test_accuracy
    )
    logging.info(
        "Selected %s with CV accuracy %.4f and test accuracy %.4f",
        artifacts.selected_model_name,
        best_result.cross_validation_accuracy,
        final_accuracy,
    )
    logging.info("Artifacts saved to %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
