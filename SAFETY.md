# Scope and limits

This project is an **educational wellness estimator**. It is not a diagnosis,
not a medical device, and makes no regulatory claim under UKCA/CE or FDA rules.

## What it does
Estimates the probability that a person is on a trajectory toward common
chronic conditions, given self-reported and wearable data, and suggests
evidence-backed changes.

## What it does not do
- Diagnose. It cannot examine, order tests, or see a full medical history.
- Handle acute presentations. Red-flag symptoms short-circuit scoring entirely
  and return a seek-care message instead (see `health_agent/safety.py`).
- Replace a clinician, or justify starting, stopping or changing medication.

## Red flags that bypass scoring
Chest pain, stroke signs (FAST), severe breathlessness, anaphylaxis, suicidal
ideation, and other acute presentations route straight to urgent-care advice.
Scoring a person who is describing an emergency is the worst failure this
system could have, so it is handled before any model is called.

## Calibration honesty
Probabilities come from a model, not from a cohort study on this population.
The eval suite (`scripts/run_evals.py`) checks ordering, monotonicity and
sensitivity -- that the numbers behave sensibly. It does **not** establish that
they are correct in absolute terms. Any absolute claim would need validation
against a real cohort with outcomes.
