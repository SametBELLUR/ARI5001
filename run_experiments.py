import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_credit_default_dataset, train_val_test_split, standardize
from src.mlp import MLPBinary
from src.metrics import classification_metrics
from src.plots import plot_train_vs_val_loss, plot_acc_vs_train_fraction, plot_noise_sensitivity
from src.baselines import logistic_regression_baseline


def train_mlp(Xtr, ytr, Xva, yva, *, hidden=32, lr=1e-3, l2=1e-4, epochs=60, batch_size=256, seed=42):
    # simple heuristic for imbalance
    pos = max(1, int((ytr == 1).sum()))
    neg = max(1, int((ytr == 0).sum()))
    pos_weight = neg / pos

    model = MLPBinary(d_in=Xtr.shape[1], h=hidden, lr=lr, l2=l2, seed=seed)

    train_loss = []
    val_loss = []
    rng = np.random.default_rng(seed)

    n = Xtr.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(n)
        Xtr_shuf = Xtr[idx]
        ytr_shuf = ytr[idx]

        # minibatches
        for i in range(0, n, batch_size):
            xb = Xtr_shuf[i:i+batch_size]
            yb = ytr_shuf[i:i+batch_size]
            p, cache = model.forward(xb)
            model.step(cache, yb, pos_weight=pos_weight)

        # epoch losses
        p_tr = model.predict_proba(Xtr)
        p_va = model.predict_proba(Xva)
        train_loss.append(model.bce(ytr, p_tr, pos_weight=pos_weight))
        val_loss.append(model.bce(yva, p_va, pos_weight=pos_weight))

    return model, np.array(train_loss), np.array(val_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, default='credit_card.csv')
    ap.add_argument('--out_dir', type=str, default='artifacts')
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--l2', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_credit_default_dataset(args.data_path)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(X, y, seed=args.seed)

    Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)

    # ---- MLP ----
    model, tr_loss, va_loss = train_mlp(
        Xtr_s, ytr, Xva_s, yva,
        hidden=args.hidden, lr=args.lr, l2=args.l2,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
    )

    plot_train_vs_val_loss(tr_loss, va_loss, out_dir / 'train_val_loss.png')

    # train fraction experiment
    fracs = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    plot_acc_vs_train_fraction(
        Xtr_s, ytr, Xva_s, yva,
        fracs=fracs,
        train_fn=lambda Xsub, ysub: train_mlp(Xsub, ysub, Xva_s, yva,
                                             hidden=args.hidden, lr=args.lr, l2=args.l2,
                                             epochs=max(20, args.epochs//2), batch_size=args.batch_size, seed=args.seed)[0],
        out_path=out_dir / 'acc_vs_train_fraction.png',
        seed=args.seed
    )

    # noise sensitivity (robustness)
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    plot_noise_sensitivity(model, Xva_s, yva, sigmas=sigmas, out_path=out_dir / 'noise_sensitivity.png', seed=args.seed)

    # test metrics
    p_te = model.predict_proba(Xte_s)
    yhat_te = (p_te >= 0.5).astype(int)
    mlp_metrics = classification_metrics(yte, yhat_te, p_te)

    (lr_metrics, lr_artifacts) = logistic_regression_baseline(Xtr_s, ytr, Xva_s, yva, Xte_s, yte)

    (out_dir / 'metrics.json').write_text(json.dumps(mlp_metrics, indent=2), encoding='utf-8')
    (out_dir / 'baseline_metrics.json').write_text(json.dumps(lr_metrics, indent=2), encoding='utf-8')

    # write a tiny README of generated artifacts
    (out_dir / 'ARTIFACTS.txt').write_text(
        "Generated:\n"
        "- train_val_loss.png\n"
        "- acc_vs_train_fraction.png\n"
        "- noise_sensitivity.png\n"
        "- metrics.json\n"
        "- baseline_metrics.json\n",
        encoding='utf-8'
    )

    print('Done. See artifacts/ for plots and metrics.')


if __name__ == '__main__':
    main()
