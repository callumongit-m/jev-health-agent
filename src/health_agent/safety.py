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


@dataclass(frozen=True, slots=True)
class SymptomPattern:
    """A combination that is alarming together but unremarkable apart.

    Headaches are common. Morning vomiting is common. Headaches that wake you
    plus morning vomiting is a textbook raised-intracranial-pressure picture,
    and no single keyword catches it.

    These deliberately name what to *do*, never what someone has. A named
    diagnosis invites self-treatment and, when wrong -- which a keyword match
    often is -- causes real harm and real fear. Urgency is the actionable
    part, and it is the part this can get right.
    """

    key: str
    advice: str
    #: every group must match somewhere in the text
    all_of: tuple[tuple[str, ...], ...]
    #: what a clinician will want to hear, so the person can say it
    tell_them: str


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

SEE_SOMEONE_URGENTLY = (
    "This combination should be looked at urgently -- today if you can. "
    "Contact your GP and ask for an urgent appointment, or call 111."
)
SEE_SOMEONE_SOON = (
    "This combination is worth getting checked properly rather than waiting "
    "to see if it passes. Book a GP appointment in the next week or two."
)

#: Patterns worth flagging. Each is a combination with a recognised urgency,
#: not an attempt at a differential. The list is short on purpose: every entry
#: has to earn its place by being both catchable from plain language and
#: genuinely time-sensitive.
SYMPTOM_PATTERNS: tuple[SymptomPattern, ...] = (
    SymptomPattern(
        key="raised_intracranial_pressure",
        advice=SEE_SOMEONE_URGENTLY,
        all_of=(
            (r"headache", r"head pain"),
            (r"vomit", r"being sick", r"throwing up", r"nausea"),
        ),
        tell_them=(
            "Say explicitly whether the headache is worse in the morning or "
            "wakes you from sleep, and whether the vomiting comes without "
            "nausea first. Those two details change how it is assessed."
        ),
    ),
    SymptomPattern(
        key="hyperglycaemia",
        advice=SEE_SOMEONE_URGENTLY,
        all_of=(
            (r"thirst", r"drinking a lot", r"always drinking"),
            (r"weeing", r"urinat", r"peeing", r"toilet all"),
        ),
        tell_them=(
            "Mention any unexplained weight loss and how long this has been "
            "going on. A finger-prick glucose test takes seconds."
        ),
    ),
    SymptomPattern(
        key="possible_malignancy",
        advice=SEE_SOMEONE_URGENTLY,
        all_of=(
            (r"weight loss", r"losing weight", r"lost weight"),
            (r"night sweat", r"fever", r"tired all the time", r"exhaust"),
        ),
        tell_them=(
            "Say how much weight over how long, and that it was not "
            "intentional. Unintentional loss is what matters."
        ),
    ),
    SymptomPattern(
        key="cardiac_exertional",
        advice=SEE_SOMEONE_URGENTLY,
        all_of=(
            (r"breathless", r"short of breath", r"puffed"),
            (r"swollen (?:ankle|leg|feet|foot)",
             r"(?:ankle|leg|feet|foot)s?\s+(?:are|is|been|look)\s+swollen",
             r"swelling in (?:my )?(?:legs|ankles|feet)",
             r"wake up gasping", r"lie flat", r"pillows to sleep"),
        ),
        tell_them=(
            "Mention whether you can lie flat to sleep and how many pillows "
            "you use. That detail matters a great deal here."
        ),
    ),
    SymptomPattern(
        key="sleep_apnoea",
        advice=SEE_SOMEONE_SOON,
        all_of=(
            (r"snor", r"stop breathing", r"gasp"),
            (r"tired", r"exhaust", r"sleepy", r"falling asleep"),
        ),
        tell_them=(
            "Ask whoever you sleep near whether you stop breathing or gasp. "
            "That observation is often what gets a sleep study arranged."
        ),
    ),
)

_COMPILED_PATTERNS = tuple(
    (
        pattern,
        tuple(
            tuple(re.compile(alt, re.IGNORECASE) for alt in group)
            for group in pattern.all_of
        ),
    )
    for pattern in SYMPTOM_PATTERNS
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


def detect_patterns(*texts: str | None) -> list[SymptomPattern]:
    """Symptom combinations that are alarming together but not apart."""
    haystack = " ".join(t for t in texts if t)
    if not haystack.strip():
        return []
    return [
        pattern
        for pattern, groups in _COMPILED_PATTERNS
        if all(any(alt.search(haystack) for alt in group) for group in groups)
    ]


def pattern_guidance(patterns: list[SymptomPattern]) -> str:
    """What to do and what to tell a clinician. Never what someone has."""
    if not patterns:
        return ""
    lines = [
        "Some of what you have described is worth acting on sooner rather "
        "than later. This is about urgency, not a diagnosis -- these "
        "combinations have plenty of ordinary explanations, and working out "
        "which is a job for someone who can examine you.",
        "",
    ]
    seen: set[str] = set()
    for pattern in patterns:
        if pattern.advice not in seen:
            lines.append(pattern.advice)
            seen.add(pattern.advice)
        lines.append(f"  {pattern.tell_them}")
    return "\n".join(lines)


def advice_for(flags: list[RedFlag]) -> str:
    """One combined message, emergency advice first."""
    seen: list[str] = []
    for flag in flags:
        if flag.advice not in seen:
            seen.append(flag.advice)
    order = {EMERGENCY: 0, CRISIS: 1, URGENT_SAME_DAY: 2}
    return "\n\n".join(sorted(seen, key=lambda a: order.get(a, 9)))
