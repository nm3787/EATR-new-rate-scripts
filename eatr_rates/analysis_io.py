from __future__ import annotations

from pathlib import Path


def sorted_prefixed_dirs(parent_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [path for path in parent_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: _suffix_value(path.name, prefix),
    )


def _suffix_value(name: str, prefix: str):
    suffix = name.replace(prefix, "", 1)
    try:
        return int(suffix)
    except ValueError:
        try:
            return float(suffix)
        except ValueError:
            return suffix


def collect_completed_run_files(parent_dir: Path, run_dir_prefix: str, colvar_glob: str, log_glob: str) -> tuple[list[str], list[str]]:
    colvars: list[str] = []
    log_files: list[str] = []
    for run_dir in sorted_prefixed_dirs(parent_dir, run_dir_prefix):
        colvar = next(run_dir.glob(colvar_glob), None)
        log_file = next(run_dir.glob(log_glob), None)
        if colvar is None or log_file is None:
            continue
        colvars.append(str(colvar))
        log_files.append(str(log_file))
    if not colvars:
        raise SystemExit(f"No completed run files found under {parent_dir}")
    return colvars, log_files
