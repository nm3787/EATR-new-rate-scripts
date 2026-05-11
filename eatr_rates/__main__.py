"""Module runner for `python -m eatr_rates`."""

from __future__ import annotations

from eatr_rates.cli import rates_cmd


if __name__ == "__main__":
    raise SystemExit(rates_cmd())
