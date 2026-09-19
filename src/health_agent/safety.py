"""Acute red-flag detection.

Scoring someone who is describing an emergency is the worst thing this system
could do, so this runs before any model is called and short-circuits the graph.
Deliberately keyword-based and over-inclusive: a false positive costs one
unnecessary "seek care" message, a false negative could cost a great deal more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedFlag:
    key: str
    advice: str
    patterns: tuple[str, ...]


EMERGENCY = (
    "This needs urgent medical attention now, not a risk estimate. "
    "Call emergency services (999 in the UK, 911 in the US) or go to A&E."
)
URGENT_SAME_DAY = (
    "This should be assessed by a clinician today. Contact your GP urgently, "
    "call 111 in the UK, or attend an urgent care centre."
)
CRISIS = (
    "Please talk to someone now. In the UK, Samaritans are on 116 123, free, "
    "any time. In the US, call or text 988. If you are in immediate danger, "
    "call emergency services."
)

RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag(
        key="cardiac",
        advice=EMERGENCY,
        patterns=(
            # allow words in between: "chest feels a bit tight" is exertional
            # angina until proven otherwise, and must not slip through
            r"\bchest\b[^.!?]{0,25}\b(pain|tight|tightness|heavy|heaviness|pressure|crushing)\b",
            r"\b(pain|tightness|pressure)\b[^.!?]{0,15}\bchest\b",
            r"\bcrushing (chest|pain)\b",
            r"\bpain (radiating|spreading) (to|down|into) (my )?(left )?(arm|jaw|neck)\b",
            r"\bheart attack\b",
        ),
    ),
    RedFlag(
        key="stroke",
        advice=EMERGENCY,
        patterns=(
            r"\bface (is )?droop", r"\bslurred speech\b", r"\bcan'?t speak\b",
            r"\bsudden (weakness|numbness)\b", r"\bone side of (my )?body\b",
            r"\bstroke\b", r"\bsudden (vision loss|blurred vision)\b",
        ),
    ),
    RedFlag(
        key="respiratory",
        advice=EMERGENCY,
        patterns=(
            r"\b(can'?t|cannot|struggling to) breathe\b",
            r"\bgasping\b", r"\bblue (lips|fingers)\b",
            r"\bsevere(ly)? (short of breath|breathless)\b",
            r"\banaphyla", r"\bthroat (is )?closing\b",
        ),
    ),
    RedFlag(
        key="bleeding",
        advice=EMERGENCY,
        patterns=(
            r"\bcoughing (up )?blood\b", r"\bvomiting blood\b",
            r"\bblood in (my )?(vomit|stool)\b", r"\buncontrolled bleeding\b",
        ),
    ),
    RedFlag(
        key="neurological",
        advice=EMERGENCY,
        patterns=(
            r"\bworst headache (of my life|ever)\b", r"\bthunderclap headache\b",
            r"\bseizure\b", r"\bloss of consciousness\b", r"\bpassed out\b",
            r"\bunresponsive\b",
        ),
    ),
    RedFlag(
        key="mental_health_crisis",
        advice=CRISIS,
        patterns=(
            r"\bsuicid",
            r"\b(kill|harm|hurt) myself\b",
            r"\bend (my life|it all)\b",
            r"\bself[- ]harm",
            # cover contraction and expanded forms alike -- a missed crisis
            # cue is the most costly false negative in this module
            r"\bdo\s*n(?:'|’)?o?t\s+want\s+to\s+(live|be here|wake up)\b",
            r"(?<!not )(?<!n't )(?<!nt )\bwant to die\b",
            r"\bno reason to (live|go on)\b",
            r"\b(better off|be better) without me\b",
            r"\bnot worth living\b",
        ),
    ),
    RedFlag(
        key="sepsis",
        advice=URGENT_SAME_DAY,
        patterns=(
            r"\bvery high fever\b", r"\bconfus(ed|ion) and fever\b",
            r"\bsepsis\b", r"\brash that doesn'?t fade\b",
        ),
    ),
    RedFlag(
        key="diabetic_emergency",
        advice=URGENT_SAME_DAY,
        patterns=(
            r"\bketoacidosis\b", r"\bdka\b",
            r"\bblood sugar (is )?(very high|over 25)\b",
            r"\bfruity breath\b",
        ),
    ),
)

_COMPILED = tuple(
    (flag, tuple(re.compile(p, re.IGNORECASE) for p in flag.patterns))
    for flag in RED_FLAGS
)


def detect(*texts: str | None) -> list[RedFlag]:
    """Red flags found anywhere in the supplied free text."""
    haystack = " ".join(t for t in texts if t)
    if not haystack.strip():
        return []
    return [flag for flag, patterns in _COMPILED if any(p.search(haystack) for p in patterns)]


def advice_for(flags: list[RedFlag]) -> str:
    """One combined message, emergency advice first."""
    seen: list[str] = []
    for flag in flags:
        if flag.advice not in seen:
            seen.append(flag.advice)
    order = {EMERGENCY: 0, CRISIS: 1, URGENT_SAME_DAY: 2}
    return "\n\n".join(sorted(seen, key=lambda a: order.get(a, 9)))
