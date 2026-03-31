from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_path: Path
    target_col: str

    drop_cols: list[str] = field(default_factory=list)
    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42

    synthetic_multiplier: int = 2
    top_k_synth: int = 100
    candidate_pool_size: int = 200
    influence_metric: str = "roc_auc"

    gc_default_distribution: str = "beta"
    max_iter: int = 1000

    output_dir: Path = Path("results")

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 < self.val_size < 1:
            raise ValueError("val_size must be between 0 and 1.")
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1.")
        if self.synthetic_multiplier < 1:
            raise ValueError("synthetic_multiplier must be >= 1.")
        if self.top_k_synth < 1:
            raise ValueError("top_k_synth must be >= 1.")
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be >= 1.")
