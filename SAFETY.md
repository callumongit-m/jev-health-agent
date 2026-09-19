# Scope, limits, and how uncertainty is handled

This document is here because anything that tells people about their health
should be explicit about what it knows, what it does not, and what it does
when it is unsure. Vagueness on any of those is the failure mode that matters.

## What this is

A risk estimator. It takes what is known about a person, returns calibrated
probabilities for common chronic conditions, estimates life expectancy from
national mortality data adjusted for modifiable factors, and ranks the changes
that would most improve both.

It is designed to be accurate about its own uncertainty, not just about its
central estimate. A probability that does not know how confident it is cannot
be acted on responsibly.

## What this is not

**Not a diagnosis, and not a medical device.** It makes no regulatory claim
under UKCA, EU MDR, or FDA rules. It cannot examine anyone, order a test, or
see a medical history. A high probability here means *this profile resembles
people who went on to develop this* — it does not mean anyone has anything.

**Not a substitute for a clinician**, and never a reason to start, stop, or
change medication.

**Not validated against a cohort.** The probabilities come from a calibrated
classifier and the life expectancy from published national life tables with
literature-derived adjustments. The test suite verifies the numbers *behave*
correctly — they order sensibly, respond to every input that should move them,
never move the wrong way, and stay stable across runs. That is a real and
non-trivial bar, and it is not the same as demonstrating accuracy against real
outcomes. Doing that needs a longitudinal cohort with recorded endpoints.
Until that exists, treat the figures as well-reasoned estimates, not measured
truths. This is stated plainly rather than buried because the distinction
matters to anyone deciding how much weight to give the output.

## Symptoms are used for urgency, never for a diagnosis

Symptom descriptions are collected, because they are genuinely informative
and because they are what makes the acute screen work. They are deliberately
*not* used to name a condition.

Two reasons. A named diagnosis invites self-treatment, which is the concrete
harm. And a keyword match made without examining anyone is often wrong, so
naming something frightening has a real cost and poor specificity to justify
it. Urgency is the part that is both actionable and gettable-right.

So combinations that are unremarkable apart but time-sensitive together — a
headache with morning vomiting, thirst with frequent urination,
breathlessness with swollen ankles — return how soon to be seen and the
specific details worth mentioning to a clinician. Never a label. A test
asserts that no pattern's text names a condition.

## Acute symptoms bypass scoring entirely

Returning a ten-year risk percentage to someone describing chest pain would be
the worst thing this system could do. So before any model runs, free text and
reported symptoms are screened for acute presentations — cardiac, stroke,
respiratory, bleeding, neurological, sepsis, diabetic emergencies, and mental
health crisis. A match short-circuits everything and returns urgent-care
guidance instead, with no risk score at all.

The screen is deliberately over-inclusive. A false positive costs one
unnecessary "please get this looked at"; a false negative could cost far more.
It is tuned against both directions — "I do not want to live" and exertional
chest tightness must trigger; "I do not want to die young, how do I improve?"
must not — and those cases are locked into the test suite.

## How uncertainty is handled

Three rules, enforced in code rather than left to good intentions:

**Numbers are withheld unless they are earned, and "enough" depends on age.**
Under 30, lifestyle and body composition carry most of the signal. From 30,
three questions anyone can answer from memory are also needed — blood
pressure medication, ever being told your blood sugar was high, whether you
eat vegetables most days — because past that age habits alone stop telling
people apart.

No age requires a test. Roughly half of UK over-40s have not had an NHS
Health Check, so a hard blood requirement would refuse the people this is
most useful to. Blood pressure and blood markers raise the confidence ceiling
instead. Below the bar the response carries no probabilities at all, just
specific questions and the reason for them — it does not return a hedged
number with a caveat, because anything handed a number presents it as final
regardless of the caveat.

**An estimate says what it rests on.** Confidence is capped by the evidence
behind it: a lifestyle-only assessment cannot report the same certainty as
one backed by bloods, however sure the classifier sounds.

**Missing data cannot become a finding.** Every factor is weighted by how
confident the classifier is. Something never measured cannot deduct years from
a life expectancy estimate, and cannot be presented as someone's top priority.

**Overlapping risks are not summed.** Smoking, poor diet and inactivity share
a great deal of their harm. Adding their individual costs together overstates
the recoverable benefit several times over, so a saturating model is applied
and the reported total is always less than the sum of its parts.

## Data handling

Health data is not being used to train any model. It is sent only to the inference
APIs required to answer a request, under their no-training terms. Nothing is
retained by default — assessments live in memory for the session and are gone
when it ends. Persistence happens only if someone opts into reminders or an
ongoing wearable sync, carries an expiry, and can be erased at any time with
`delete_my_data`. Values are redacted from logs by field name.

## Reporting a problem

If you find a case where this gives dangerous advice, misses an acute
presentation, or produces a number it cannot justify, please open an issue.
Those are the bugs worth hearing about.
