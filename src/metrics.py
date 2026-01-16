import numpy as np


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = _safe_div(tp + tn, len(y_true))
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    out = {
        'n': int(len(y_true)),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

    # optional approximate AUC if probabilities are provided
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        out['auc_approx'] = _roc_auc_approx(y_true, y_prob)

    return out


def _roc_auc_approx(y_true, y_prob):
    # fast rank-based AUC approximation
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0

    # pairwise comparison approximation (can be slow on huge arrays; sample for speed)
    rng = np.random.default_rng(42)
    m = min(5000, len(pos))
    n = min(5000, len(neg))
    pos_s = rng.choice(pos, size=m, replace=False) if len(pos) > m else pos
    neg_s = rng.choice(neg, size=n, replace=False) if len(neg) > n else neg

    # AUC = P(score_pos > score_neg) + 0.5 P(equal)
    cmp = pos_s[:, None] - neg_s[None, :]
    return float((cmp > 0).mean() + 0.5 * (cmp == 0).mean())
