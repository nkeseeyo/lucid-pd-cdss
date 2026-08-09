"""Person-level cross-validation — the project's central anti-leakage control.

The single most common cause of inflated accuracy in this literature is letting
recordings from the same speaker fall on both sides of the train/test split. We
ALWAYS group on the subject id so no individual is ever in both partitions, and
we fit every preprocessing step INSIDE each training fold.

This module is intentionally a thin, well-tested wrapper around scikit-learn so
the methodology is explicit and auditable in the dissertation.
"""
from __future__ import annotations

# NOTE: implementation lands in Week 1 (see PROJECT_PLAN.md). Signature is fixed
# now so downstream code (models, evaluate, fusion) can be written against it.


def grouped_cv_iter(X, y, groups, n_splits: int = 5, seed: int = 42):
    """Yield (train_idx, test_idx) using sklearn.model_selection.GroupKFold.

    Parameters
    ----------
    X, y : feature matrix and binary target (1 = PD, 0 = HC).
    groups : subject id per row (UCI #470 'id'); guarantees no speaker leakage.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    yield from splitter.split(X, y, groups)


def naive_record_cv_iter(X, y, n_splits: int = 5, seed: int = 42):
    """DELIBERATELY-WRONG record-level split — used ONLY for the leakage demo
    figure (RQ1): it inflates accuracy because the same speaker leaks across
    folds. Never use for reported results.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(X, y)
