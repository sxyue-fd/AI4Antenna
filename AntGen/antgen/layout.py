"""Stable on-disk layout for one AntGen run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def surrogate(self) -> Path:
        return self.root / "surrogate"

    @property
    def fullwave(self) -> Path:
        return self.root / "fullwave"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def manifest(self) -> Path:
        return self.artifacts / "manifest.json"

    @property
    def candidates(self) -> Path:
        return self.artifacts / "generated_candidates.npz"

    @property
    def surrogate_predictions(self) -> Path:
        return self.surrogate / "predictions.npz"

    @property
    def surrogate_ranking(self) -> Path:
        return self.surrogate / "candidate_ranking.csv"

    @property
    def surrogate_best(self) -> Path:
        return self.surrogate / "best_per_condition.csv"

    @property
    def surrogate_summary(self) -> Path:
        return self.surrogate / "summary.json"

    @property
    def selected_candidates(self) -> Path:
        return self.fullwave / "selected_candidates.npz"

    @property
    def matlab_input(self) -> Path:
        return self.fullwave / "matlab_input.mat"

    @property
    def matlab_output(self) -> Path:
        return self.fullwave / "matlab_output.mat"

    @property
    def fullwave_results(self) -> Path:
        return self.fullwave / "results.csv"

    @property
    def fullwave_summary(self) -> Path:
        return self.fullwave / "summary.json"

    @property
    def final_summary(self) -> Path:
        return self.reports / "summary.json"

    def ensure(self) -> "RunLayout":
        for directory in (
            self.root,
            self.artifacts,
            self.surrogate,
            self.fullwave,
            self.reports,
            self.figures,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def run_layout(run_dir: Path, create: bool = False) -> RunLayout:
    layout = RunLayout(Path(run_dir).resolve())
    return layout.ensure() if create else layout
