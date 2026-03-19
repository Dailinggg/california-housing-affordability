# California Housing Affordability — Reproducible Analysis

Reproducible statistical analysis of California housing values (1990 census), focusing on income effects and ocean proximity.

**Report (PDF):** `report/california_housing_affordability_public.pdf`

## At-a-glance results (matches the report)
- Correlation r(median_income, median_house_value) ≈ **0.69**
- Multiple regression R² ≈ **0.6076** (F ≈ **3993.81**)
- One-way ANOVA across ocean proximity: F ≈ **1612.14**, η² ≈ **0.238**
- K-means (k=4) silhouette ≈ **0.315**
- 5-fold CV R² mean ≈ **0.6433**

## Key figures
**Figure 4 — Geographic clustering**
![Figure 4](outputs/figures/figure_4_geographic_clusters.png)

**Figure 1 — Correlation matrix**
![Figure 1](outputs/figures/figure_1_correlation_matrix.png)

## Quickstart (local)
```bash
pip install -r requirements.txt
# place housing.csv in repo root (see data/README.md)
python src/california_housing_affordability_analysis.py
