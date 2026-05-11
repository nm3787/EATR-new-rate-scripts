from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="the simulation COLVAR files to check the order of", nargs="+")
    parser.add_argument("-l", "--logfiles", type=str, default=None, help="the simulation PLUMED log files to check the order of", nargs="+")
    parser.add_argument("-o", "--output", type=str, default="order.dat", help="the name of the output file which will contain the COLVAR files in order of Unix unpacking")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.logfiles is not None and len(args.input) != len(args.logfiles):
        raise SystemExit("There are an unequal number of COLVAR files and PLUMED log files.")

    with open(args.output, "w", encoding="utf-8") as handle:
        if args.logfiles is None:
            for colvar in args.input:
                handle.write(colvar + "\n")
        else:
            for colvar, logfile in zip(args.input, args.logfiles):
                handle.write(colvar + " | " + logfile + "\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
