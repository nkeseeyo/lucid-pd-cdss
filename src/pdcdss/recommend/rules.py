"""Risk-band -> care-route rules. Transparent, deterministic, unit-testable.

Design choices (defensible in the viva):
  * Decision SUPPORT, not diagnosis — the route is a suggestion for a clinician.
  * Specialist *types* only (neurologist with movement-disorder interest, PD nurse
    specialist, physiotherapy, OT, SALT) — never real, named individuals.
  * Sensitivity-weighted: when in doubt, escalate.

Risk bands follow the 0-40 / 40-70 / 70-100 % split used throughout the UI.
"""
from __future__ import annotations

from dataclasses import dataclass

# illustrative specialist TYPES only (NICE NG71-aligned), never named people
SPECIALIST_TYPES = [
    "Neurologist (movement-disorder interest)",
    "Parkinson's disease nurse specialist",
    "Physiotherapy",
    "Occupational therapy",
    "Speech and language therapy",
]

_DISCLAIMER = (
    "This tool supports clinical judgement and does not replace a full clinical assessment."
)


@dataclass(frozen=True)
class CareRoute:
    band: str
    route: str
    specialist_type: str
    secondary: str = ""
    disclaimer: str = _DISCLAIMER


def risk_to_band(probability: float) -> str:
    """Map a calibrated PD probability to a 3-level risk band."""
    if probability < 0.40:
        return "low"
    if probability < 0.70:
        return "moderate"
    return "high"


_TABLE = {
    "low": CareRoute("low", "Routine care and reassurance", "Primary care (GP) review"),
    "moderate": CareRoute(
        "moderate", "Refer for specialist assessment",
        "Neurologist (movement-disorder interest)", "Parkinson's nurse specialist"),
    "high": CareRoute(
        "high", "Expedited specialist referral",
        "Neurologist (movement-disorder interest)", "Parkinson's nurse specialist"),
}


def recommend(probability: float) -> CareRoute:
    return _TABLE[risk_to_band(probability)]
