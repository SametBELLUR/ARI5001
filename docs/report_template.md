# ARI 5001 Take-home Project — Theme 5 Report Template

> Replace all `<<PLACEHOLDER>>` blocks with your results and your own writing.

## Title
<<Your title>>

## I. Problem Statement
- Dataset: D = {(x_i, y_i)} where y ∈ {0,1} indicates default.
- Task: Binary classification to predict default.
- Objective: Learn f_θ(x) minimizing empirical risk using weighted BCE.
- Evaluation: Accuracy, Precision, Recall, F1 (and optional AUC).

## II. Theoretical Framework
### Model
- 1-hidden-layer MLP: h = ReLU(W1 x + b1), p = sigmoid(W2 h + b2)

### Loss
- Weighted Binary Cross Entropy:
  L = -[w+ y log p + (1-y) log(1-p)]

### Optimizer
- Mini-batch SGD (describe your learning rate, batch size, epochs).

### Pseudocode (Write in your own words)
<<YOUR PSEUDOCODE>>

### Complexity (Big-O)
Let N be samples, d input dim, h hidden dim, E epochs.
- Forward+backprop per batch is O(batch * d*h + batch*h)
- Full epoch O(N*d*h)
- Total O(E*N*d*h)
Space O(d*h + h)

## III. Implementation
- Data split strategy (train/val/test; stratified).
- Standardization.
- Any preprocessing choices (e.g., treating categorical columns as numeric).

## IV. Experiments & Results (Theme 5 requirements)
### A) Train vs Validation Loss
Insert plot: `artifacts/train_val_loss.png`
- Observation: <<what happens to train vs val loss?>>
- Interpretation: <<overfitting? underfitting? why?>>

### B) Accuracy vs Training Set Size
Insert plot: `artifacts/acc_vs_train_fraction.png`
- Observation: <<sample inefficiency trends>>

### C) Sensitivity to Controlled Perturbations
Insert plot: `artifacts/noise_sensitivity.png`
- Limitation discussed: robustness under domain shift / perturbations

### D) Baseline Comparison
Use `artifacts/metrics.json` (MLP) and `artifacts/baseline_metrics.json` (LogReg)
- Table of metrics: <<fill from json>>
- Discussion: <<why MLP differs from baseline>>

### E) Parameter Sensitivity
Pick one: hidden size sweep or lr sweep (add your plot/table)
- Conclusion: <<which hyperparameter matters most and why?>>

## V. AI Critique (Strongly weighted)
Include one AI response you used during development.
- Prompt: <<your prompt>>
- Response summary: <<short summary>>
- Identified error: <<specific mistake>>
- Correction: <<corrected explanation>>
- Compare to course theory: <<what the error shows about AI limits>>

## VI. AI-Assisted Development Log (Appendix)
See `docs/ai_dev_log_template.md` and attach it.

## VII. Academic Integrity Statement
Paste the exact statement required by your assignment PDF.

## References
- Russell & Norvig, *Artificial Intelligence: A Modern Approach* (relevant sections)
- Dataset source (UCI / Kaggle etc.)
