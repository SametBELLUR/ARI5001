from __future__ import annotations

import json
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import classification_metrics


def logistic_regression_baseline(Xtr, ytr, Xva, yva, Xte, yte) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # class_weight balanced is a common baseline for imbalanced labels
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=1)
    clf.fit(Xtr, ytr)

    p_te = clf.predict_proba(Xte)[:, 1]
    yhat_te = (p_te >= 0.5).astype(int)

    metrics = classification_metrics(yte, yhat_te, p_te)

    artifacts = {
        'coef': clf.coef_.reshape(-1).tolist(),
        'intercept': float(clf.intercept_[0]),
    }

    return metrics, artifacts
