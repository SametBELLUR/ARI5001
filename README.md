# ARI 5001 Take-home Project — Theme 5 Scaffold (MLP)

This repository is a **Theme 5 (Deep Learning Systems)** scaffold based on the UCI credit default dataset (`credit_card.csv`).

**What this gives you**
- A small **MLP (from scratch, NumPy)** for binary classification
- Theme-5-required experiments:
  1) train vs validation loss curve
  2) accuracy vs training set size
  3) controlled perturbation sensitivity (noise) curve
- A baseline **Logistic Regression** comparison
- Clean, reproducible scripts that generate plots + metrics under `artifacts/`
- A **report template** + **AI development log template** aligned with the assignment rubric

**What you still must do (academically)**
- Run the experiments, collect results, and **write your own interpretations** in the report.
- Fill in the **AI Critique**, **AI-assisted development log**, and the exact **Academic Integrity Statement** required by your course handout.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run all experiments

From the project root:

```bash
python run_experiments.py --data_path ../credit_card.csv
```

Outputs:
- `artifacts/train_val_loss.png`
- `artifacts/acc_vs_train_fraction.png`
- `artifacts/noise_sensitivity.png`
- `artifacts/metrics.json`
- `artifacts/baseline_metrics.json`

You can also tweak hyperparameters:

```bash
python run_experiments.py --data_path ../credit_card.csv --hidden 32 --lr 0.001 --epochs 60
```

---

## Report

Use `docs/report_template.md` and replace the placeholders with:
- your plots (saved in `artifacts/`)
- your measured metrics (from `artifacts/metrics.json`)
- your own discussion

Also fill:
- `docs/ai_dev_log_template.md`

---

## Notes
- The dataset has a column named `default payment_next_month` (1 = default). The code treats **default=1** as the positive class.
- This scaffold standardizes all features. (You may optionally one-hot encode categorical fields; document any changes.)
