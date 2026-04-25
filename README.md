# Linear Algebros — Public Transit Usage Prediction

**Course project** · Spring 2026

**Group members**
| Name | Section |
|---|---|
| Devki Misra | PCA |
| Ignacio Montoro Piñeiro | PCA |
| Lucian Mikush | Laplacian Eigenmaps |
| Michael Andrade Maldonado | Laplacian Eigenmaps |
| Miguel Cuevas | Supervised Learning |

---

## Project Overview

We predict the percentage of commuters using public transportation across 35 major U.S. cities using socioeconomic, demographic, and urban characteristics drawn from the [Big Cities Health Inventory](https://bigcitieshealthdata.org/) (BCHI) dataset.

**Research question:** What city-level factors best explain and predict public transit usage, and do cities cluster meaningfully by those factors?

---

## Repository Structure

```
linear-algebros/
├── data/
│   ├── BigCitiesHealth.csv          # raw BCHI dataset (original)
│   └── cities_features_2019.csv     # cleaned wide-format pivot (2019 snapshot)
├── pca/                             # Devki & Ignacio — R
│   ├── pca_analysis.Rmd
│   └── outputs/
├── laplacian/                       # Lucian & Michael — R
│   ├── laplacian_eigenmaps.Rmd
│   └── outputs/
├── supervised/                      # Miguel — Python
│   ├── supervised_learning.py
│   └── outputs/
├── report/                          # final combined writeup
│   └── final_report.Rmd
├── linear-algebros.Rproj            # open this in RStudio
├── renv.lock                        # R package lockfile (renv)
├── requirements.txt                 # Python package lockfile
└── README.md
```

---

## Data

**Source:** [Big Cities Health Inventory](https://bigcitieshealthdata.org/) — 35 major U.S. cities, 100+ health and socioeconomic metrics.

**Target variable:** `Public Transportation Use` — percentage of commuters using public transit (2019).

**Features used (24 total):**

| Feature | Notes |
|---|---|
| Population Density | Strongest predictor (r = 0.95) |
| Walking to Work | Proxy for walkability |
| Riding Bike to Work | Proxy for bikeability |
| Longer Driving Commute Time | |
| Per-capita Household Income | |
| College Graduates | |
| Foreign Born Population | |
| Income Inequality | |
| Racial Segregation (White/Non-White) | |
| Renters vs. Owners | |
| Excessive Housing Cost | |
| Limited Internet Access | |
| Limited Supermarket Access | |
| People with Disabilities | |
| Unemployment | |
| Public Assistance | |
| Service Workers | |
| Uninsured (All Ages) | |
| Adult Physical Inactivity | |
| Life Expectancy | |
| Homicides | |
| Violent Crime | |
| Poor Air Quality | |
| Longer Summers | |

The cleaned, wide-format dataset (`data/cities_features_2019.csv`) is the starting point for all three sections.

---

## Sections

### 1. PCA & Descriptive Visualization (`pca/`) — Devki & Ignacio
- Project cities onto top 2 principal components
- Color points by public transit usage percentage
- Interpret loadings to identify which features drive the PCs

### 2. Laplacian Eigenmaps (`laplacian/`) — Lucian & Michael
- Spectral dimensionality reduction using graph Laplacian
- Experiment with different kernel functions (Gaussian, k-NN graph)
- Compare embedding structure to PCA

### 3. Supervised Learning (`supervised/`) — Miguel
- **Models:** OLS, Ridge, Lasso, KNN, Random Forest
- **Evaluation:** Leave-One-Out CV (LOO) — chosen because n = 35
- **Best model:** Ridge regression (LOO-R² = 0.801, LOO-RMSE = 4.18 pp)

---

## Reproducing Results

### Python (Miguel's section)
```bash
pip install -r requirements.txt
cd supervised
python supervised_learning.py
# outputs saved to supervised/outputs/
```

### R (Devki, Ignacio, Lucian, Michael)
Open `linear-algebros.Rproj` in RStudio, then:
```r
renv::restore()   # installs all R packages from renv.lock
```
Then knit the relevant `.Rmd` file.

---

## Key Findings

- **Population density** is by far the strongest predictor of transit use (r = 0.95)
- **Northeast cities** (NYC, Boston, DC, SF) cluster at high transit use; **Southern and Western cities** cluster at low use — consistent with our hypothesis
- **Ridge regression** outperforms all other models under LOO-CV, confirming that regularization is essential with n = 35 and p = 24
- Tree-based methods (Random Forest) underperform on this small dataset
