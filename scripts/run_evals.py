#!/usr/bin/env python
"""Run the eval suite and report pass^k."""

from __future__ import annotations

import argparse
import sys

from evals.runner import format_report, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="trials per check")
    parser.add_argument("--suite", default="jev", choices=["jev", "graph", "all"])
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "jev", "openrouter", "fake"],
        help="auto uses Jev when TYPESAFE_API_KEY is set, else the offline fake",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--min-pass-k",
        type=float,
        default=None,
        help="exit non-zero if pass^k falls below this (for CI)",
    )
    args = parser.parse_args()

    report = run(k=args.k, suite=args.suite, backend=args.backend, seed=args.seed)
    print(format_report(report, verbose=args.verbose))

    if args.min_pass_k is not None and report.pass_hat_k < args.min_pass_k:
        print(f"\nFAIL: pass^k {report.pass_hat_k:.1%} < {args.min_pass_k:.1%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
