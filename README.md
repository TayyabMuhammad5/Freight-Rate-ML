# Freight Rate Prediction Challenge — Solution Repository

This repository contains the complete machine learning solution, validation suite, predictions, and technical report for the **Spotter Freight Rate Prediction Challenge**.

## Quick Start & Reproducibility

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run Training, Validation & Final Prediction Generation
```bash
python train_and_predict.py
```
Running `train_and_predict.py` executes:
1. **Data Quality Remediation & Preprocessing**:
   - Removes `quote_signal` (target leakage in M1,2,3,6,9; inverted in M4,5,7,10; pure Gaussian noise in M8 and Validation M11–12; absent in `december_chart_inputs.csv`).
   - Filters the 677 corrupted target outliers (`posted_rate / distance < $1.35/mi` or `> $3.80/mi`, 1.41% of `train_test.csv`), retaining the 47,323 clean loads.
   - Fixes sign-flipped negative weights (`weight < 0` on 292 train and 145 validation rows) via `abs(weight)` and imputes missing weights with the training median (`31,493.5 lb`) plus a missingness indicator (`wt_na`).
   - Decomposes noisy/missing `market_index` into the denoised daily market index (`daily_mkt`) and within-day deviation (`mkt_diff`), resolving errors-in-variables attenuation bias and providing exact daily market indices for December 2025.
2. **Multi-Tier Validation**:
   - **2-Month Temporal Holdout** (Train: Jan–Aug 2025 -> Validate: Sep–Oct 2025): Achieves **$40.73 Rate RMSE** (**1.197% MAPE**, **$R^2 = 0.999123$**, **$0.03365/mi RPM RMSE**, **$R^2_{\text{RPM}} = 0.98491$**), compared to $191.13 Rate RMSE for a naive baseline.
   - **5-Fold Expanding-Window Monthly Temporal CV** (Months 6–10): Achieves consistent $39.15–$44.94 Rate RMSE ($1.185%–1.284% MAPE) across all seasonal regimes.
   - **Spatio-Temporal Unseen-City Holdout**: Validates 2D Gaussian Radial Basis Function (`RBFInterpolator`) spatial interpolation across held-out cities (Leave-One-City-Out $R^2 = 0.9957$).
3. **Final Inference**:
   - Generates `validation_predictions.csv` (`load_id,predicted_rate`, 12,000 rows matching `validation_predictions_template.csv`).
   - Populates `data/december_chart_inputs.csv` (`predicted_rate` filled for all 31 days of December 2025 on the fixed `Lexington -> Fort Wayne` Dry Van corridor).

### 3. Run the Official Scorer & Generate December Chart
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
This validates both prediction files and generates `scorer_results/candidate_december.png`.

## Repository Contents
- `train_and_predict.py` — End-to-end data cleaning, feature engineering, validation, and inference script.
- `validation_predictions.csv` — Final predictions for all 12,000 validation loads (`TE-000001` to `TE-012000`).
- `data/december_chart_inputs.csv` — Completed 31-day December 2025 predictions for `Lexington -> Fort Wayne`.
- `scorer_results/candidate_december.png` — Fixed December 2025 prediction chart produced by `score.py`.
- `freight_rate_ml_report.pdf` — Technical report detailing EDA findings, validation design, model architecture, and December chart analysis.
- `loom_walkthrough_script.md` — Structured 2–3 minute Loom walkthrough script covering all required talking points.
- `requirements.txt` — Python package dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`).
- `score.py` — Provided evaluation and chart generation script.
