from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_toml(config_path: Path) -> tuple[dict, Path]:
    resolved = config_path.resolve()
    with open(resolved, "rb") as handle:
        data = tomllib.load(handle)
    return data, resolved.parent


@dataclass(frozen=True)
class CommonAnalysisConfig:
    config_path: Path
    input_root: Path
    output_root: Path
    timeunit_seconds: float
    temperature_k: float
    energyunit_kj_per_mol: float
    bootstrap_resamples: int
    threads_env_var: str

    @property
    def default_threads(self) -> int:
        return max(1, int(os.environ.get(self.threads_env_var, "1")))


@dataclass(frozen=True)
class OpesAnalysisConfig(CommonAnalysisConfig):
    bias_col: int
    colvar_glob: str
    log_glob: str
    cv_dir_prefix: str
    barrier_dir_prefix: str
    run_dir_prefix: str
    max_barrier_included: float


@dataclass(frozen=True)
class ImetadAnalysisConfig(CommonAnalysisConfig):
    bias_col: int
    acc_col: int
    colvar_glob: str
    log_glob: str
    cv_dir_prefix: str
    height_dir_prefix: str
    pace_dir_prefix: str
    run_dir_prefix: str
    timestep_ps: float


def load_opes_config(config_path: Path) -> OpesAnalysisConfig:
    raw, base_dir = _load_toml(config_path)
    analysis = raw["analysis"]
    filesystem = raw["filesystem"]
    columns = raw["columns"]
    filtering = raw["filtering"]
    return OpesAnalysisConfig(
        config_path=config_path.resolve(),
        input_root=_resolve_path(base_dir, analysis["input_root"]),
        output_root=_resolve_path(base_dir, analysis["output_root"]),
        timeunit_seconds=float(analysis["timeunit_seconds"]),
        temperature_k=float(analysis["temperature_k"]),
        energyunit_kj_per_mol=float(analysis["energyunit_kj_per_mol"]),
        bootstrap_resamples=int(analysis["bootstrap_resamples"]),
        threads_env_var=str(analysis.get("threads_env_var", "EATR_THREADS")),
        bias_col=int(columns["bias_col"]),
        colvar_glob=str(filesystem["colvar_glob"]),
        log_glob=str(filesystem["log_glob"]),
        cv_dir_prefix=str(filesystem.get("cv_dir_prefix", "")),
        barrier_dir_prefix=str(filesystem.get("barrier_dir_prefix", "barrier")),
        run_dir_prefix=str(filesystem.get("run_dir_prefix", "s")),
        max_barrier_included=float(filtering["max_barrier_included"]),
    )


def load_imetad_config(config_path: Path) -> ImetadAnalysisConfig:
    raw, base_dir = _load_toml(config_path)
    analysis = raw["analysis"]
    filesystem = raw["filesystem"]
    columns = raw["columns"]
    return ImetadAnalysisConfig(
        config_path=config_path.resolve(),
        input_root=_resolve_path(base_dir, analysis["input_root"]),
        output_root=_resolve_path(base_dir, analysis["output_root"]),
        timeunit_seconds=float(analysis["timeunit_seconds"]),
        temperature_k=float(analysis["temperature_k"]),
        energyunit_kj_per_mol=float(analysis["energyunit_kj_per_mol"]),
        bootstrap_resamples=int(analysis["bootstrap_resamples"]),
        threads_env_var=str(analysis.get("threads_env_var", "EATR_THREADS")),
        bias_col=int(columns["bias_col"]),
        acc_col=int(columns["acc_col"]),
        colvar_glob=str(filesystem["colvar_glob"]),
        log_glob=str(filesystem["log_glob"]),
        cv_dir_prefix=str(filesystem.get("cv_dir_prefix", "")),
        height_dir_prefix=str(filesystem.get("height_dir_prefix", "height")),
        pace_dir_prefix=str(filesystem.get("pace_dir_prefix", "pace")),
        run_dir_prefix=str(filesystem.get("run_dir_prefix", "s")),
        timestep_ps=float(analysis["timestep_ps"]),
    )
