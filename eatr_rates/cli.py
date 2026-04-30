"""Console entry points for the packaged analysis tools."""

from __future__ import annotations

from eatr_rates.check_order import main as check_order_main
from eatr_rates.rates_cmd import main as rates_cmd_main
from eatr_rates.rates_eatr_opes import main as rates_eatr_opes_main


def rates_cmd() -> int:
    return rates_cmd_main()


def rates_eatr_opes() -> int:
    return rates_eatr_opes_main()


def check_order() -> int:
    return check_order_main()
