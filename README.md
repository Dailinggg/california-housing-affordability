# California Housing Affordability — Reproducible Analysis

A reproducible statistical analysis of California housing values (1990 census), focusing on income effects and ocean proximity.

## Key results (matches the report)
- Correlation r(median_income, median_house_value) ≈ 0.69
- Multiple regression R² ≈ 0.6076 (F ≈ 3993.81)
- One-way ANOVA across ocean proximity: F ≈ 1612.14, η² ≈ 0.238
- K-means (k=4) silhouette ≈ 0.315
- 5-fold CV R² mean ≈ 0.6433

## Repository structure
- eport/ public PDF report
- src/ analysis script
- outputs/figures/ generated figures (Figure 1–5)
- outputs/tables/ generated tables (Table 1–7)
- data/ instructions to obtain the dataset (raw data not redistributed)

## How to run (local)
1. Install dependencies:
   pip install -r requirements.txt
2. Download housing.csv (see data/README.md) and place it in the repo root:
   ./housing.csv
3. Run:
   python src/california_housing_affordability_analysis.py

## Data
This repo does not redistribute the raw dataset. See data/README.md for download instructions.

## Outputs
Figures and tables are written to outputs/.
