#!/usr/bin/env python
"""Report what is configured, and what a run would actually use.

Spends nothing by default. Pass --live to make one real Jev call and one
real reasoning call so you know the keys work, not just that they are set.
"""

from __future__ import annotations

import argparse
import os
import sys

OK, WARN, BAD = "  ok ", " info", " MISS"


def _mask(value: str | None) -> str:
    if not value:
        return "not set"
    return f"{value[:7]}...{value[-4:]}" if len(value) > 14 else "set"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true",
        help="make one real call to each configured provider (costs a fraction of a penny)",
    )
    args = parser.parse_args()

    from health_agent.config import SETTINGS  # loads .env on import
    from health_agent.scoring.backend import get_backend

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    print(f"{OK if env_file.exists() else BAD}  .env  {env_file}")
    if not env_file.exists():
        print("        create it with:  cp .env.example .env")

    print("\nKEYS")
    typesafe = os.getenv("TYPESAFE_API_KEY")
    openrouter = os.getenv("OPENROUTER_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    print(f"{OK if typesafe else WARN}  TYPESAFE_API_KEY    {_mask(typesafe)}")
    print(f"{OK if openrouter else WARN}  OPENROUTER_API_KEY  {_mask(openrouter)}")
    if not (typesafe or openrouter):
        print(f"{BAD}  no Jev key -- scoring will use the offline fake")
    print(f"{OK if anthropic_key else BAD}  ANTHROPIC_API_KEY   {_mask(anthropic_key)}")
    if not anthropic_key:
        print("        reasoning will use the offline stand-in")

    print("\nWHAT A RUN WOULD USE")
    backend = get_backend("auto", seed=0)
    print(f"       scoring    {backend.name}")
    print(f"       reasoning  "
          f"{SETTINGS.reasoning_model if anthropic_key else 'offline stand-in'}")
    print(f"       websearch  "
          f"{SETTINGS.web_search_tool['type'] if SETTINGS.enable_web_search else 'disabled'}")

    if not args.live:
        print("\nrun with --live to prove the keys actually work")
        return 0

    print("\nLIVE CHECKS")
    from health_agent.domain.profile import HealthProfile, Sex

    probe = HealthProfile(age=54, sex=Sex.MALE, height_cm=178, weight_kg=98,
                          hba1c_mmol_mol=46, systolic_bp=148, diastolic_bp=92)

    if typesafe or openrouter:
        try:
            from health_agent.scoring.scorer import RiskScorer

            live = get_backend("jev" if typesafe else "openrouter")
            assessment = RiskScorer(live).score(probe)
            print(f"{OK}  {live.name}  ->  t2d {assessment.condition('t2d_10yr').probability:.1%}"
                  f", {assessment.latency_ms}ms")
        except Exception as exc:
            print(f"{BAD}  Jev call failed: {type(exc).__name__}: {exc}")
            return 1

    if anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic

            reply = ChatAnthropic(
                model=SETTINGS.reasoning_model, max_tokens=16
            ).invoke("Reply with the single word: ready")
            print(f"{OK}  {SETTINGS.reasoning_model}  ->  {reply.content!r}")
        except Exception as exc:
            print(f"{BAD}  Anthropic call failed: {type(exc).__name__}: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
