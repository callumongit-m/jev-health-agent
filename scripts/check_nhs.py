#!/usr/bin/env python
"""Probe the NHS Content API and report exactly what came back.

The two NHS platforms differ in host and auth header, and both answer 401
for a key meant for the other, so a failure is easy to misread. This says
which one your key works against and whether the parser understood the
response.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

CANDIDATES = [
    ("NHS England platform (Apigee)", "https://api.service.nhs.uk/nhs-website-content", "apikey"),
    ("legacy portal (Azure APIM)", "https://api.nhs.uk", "subscription-key"),
    ("sandbox, no key needed", "https://sandbox.api.service.nhs.uk/nhs-website-content", None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="conditions/type-2-diabetes")
    args = parser.parse_args()

    import httpx

    from health_agent.evidence.sources import _extract_points

    api_key = os.getenv("NHS_API_KEY", "").strip()
    print(f"NHS_API_KEY: {'set' if api_key else 'NOT SET'}\n")

    working: list[str] = []
    for label, base, header in CANDIDATES:
        if header and not api_key:
            print(f"{'skip':>6}  {label}: needs a key")
            continue
        headers = {"Accept": "application/json"}
        if header:
            headers[header] = api_key
        try:
            response = httpx.get(
                f"{base}/{args.slug}", headers=headers, timeout=15.0
            )
        except Exception as exc:
            print(f"{'err':>6}  {label}: {type(exc).__name__}")
            continue

        print(f"{response.status_code:>6}  {label}  ({header or 'no auth'})")
        if response.status_code != 200:
            detail = response.text[:120].replace("\n", " ")
            print(f"        {detail}")
            continue

        working.append(base)
        try:
            body = response.json()
        except Exception:
            print("        200 but not JSON")
            continue

        points = _extract_points(body)
        print(f"        top-level keys: {sorted(body)[:10]}")
        if points:
            print(f"        parser understood it -- {len(points)} points:")
            for point in points:
                print(f"          - {point[:88]}")
        else:
            print("        PARSER DID NOT UNDERSTAND THIS SHAPE.")
            print("        Send this to fix the mapping:")
            print(json.dumps(body, indent=2)[:1200])

    print()
    if working:
        print(f"Set NHS_API_BASE={working[0]}")
    else:
        print("Nothing answered. Register at digital.nhs.uk/developer for the")
        print("NHS Website Content API v2, then set NHS_API_KEY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
