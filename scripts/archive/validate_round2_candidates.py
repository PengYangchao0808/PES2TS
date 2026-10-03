"""Compatibility entry for gradient-ranked, independent saved-seed validation.

Round-two inputs are read-only. New receipts live under round three; partial
prefixes are eligible. Use validate_round3_candidates.py for controls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from validate_round3_candidates import validate as validate_seed


def validate(rid, method="GFN2-xTB"):
    return validate_seed(rid, source="round2", method=method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--method", choices=["GFN2-xTB", "HF-3c"], default="GFN2-xTB")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda rid: validate(rid, args.method), args.ids))


if __name__ == "__main__":
    main()
