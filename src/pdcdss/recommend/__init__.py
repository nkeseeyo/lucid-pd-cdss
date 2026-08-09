"""Rule-based care-route recommender (safe-by-design).

NOT a list of real named clinicians. Maps a predicted risk band to an
evidence-based care route and an illustrative specialist *type*, aligned with
NICE NG71 referral guidance. Fully transparent and unit-testable.

See recommend/rules.py for the rule table and rationale.
"""
