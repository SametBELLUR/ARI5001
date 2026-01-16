from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_train_vs_val_loss(train_loss, val_loss, out_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('epoch')
    plt.ylabel('BCE loss')
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_acc_vs_train_fraction(Xtr, ytr, Xva, yva, fracs, train_fn, out_path, seed=42):
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    accs = []

    for f in fracs:
        k = max(200, int(f * n))
        idx = rng.choice(n, size=k, replace=False)
        model = train_fn(Xtr[idx], ytr[idx])
        yhat = model.predict(Xva)
        acc = float((yhat == yva).mean())
        accs.append(acc)

    plt.figure()
    plt.plot(fracs, accs, marker='o')
    plt.xlabel('training fraction')
    plt.ylabel('validation accuracy')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_noise_sensitivity(model, Xva, yva, sigmas, out_path, seed=42):
    rng = np.random.default_rng(seed)
    base = float((model.predict(Xva) == yva).mean())

    accs = []
    for s in sigmas:
        Xn = Xva + rng.normal(0, s, size=Xva.shape)
        accs.append(float((model.predict(Xn) == yva).mean()))

    plt.figure()
    plt.plot(sigmas, accs, marker='o', label='noisy')
    plt.axhline(base, linestyle='--', label='baseline')
    plt.xlabel('noise sigma')
    plt.ylabel('validation accuracy')
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
