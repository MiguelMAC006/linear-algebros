"""
Linear Algebros – Supervised Learning Module
Miguel Cuevas

Predicting Public Transportation Use (% of commuters) across 35 U.S. cities
using Linear Regression, K-Nearest Neighbors, and Random Forests.

Dataset: Big Cities Health Inventory  (bigcitieshealthdata.org)
"""

# =============================================================================
# 0. Imports
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score, KFold, GridSearchCV, LeaveOneOut
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Reproducibility
SEED = 42
np.random.seed(SEED)

# =============================================================================
# 1. Data Loading & Pivoting
# =============================================================================

RAW_PATH = "/mnt/user-data/uploads/BigCitiesHealth.csv"

# --- feature map: descriptive name → exact metric_item_label in dataset ---
FEATURE_MAP = {
    "People with Disabilities":             "People with Disabilities",
    "Adult Physical Inactivity":            "Adult Physical Inactivity",
    "Life Expectancy":                      "Life Expectancy",
    "Homicides":                            "Homicides",
    "Violent Crime":                        "Violent Crime",
    "Unemployment":                         "Unemployment",
    "Public Assistance":                    "Public Assistance",
    "Service Workers":                      "Service Workers",
    "Excessive Housing Cost":               "Excessive Housing Cost",
    "Income Inequality":                    "Income Inequality",
    "Racial Segregation (White/Non-White)": "Racial Segregation, White and Non-White",
    "Limited Internet Access":              "Limited Internet Access",
    "College Graduates":                    "College Graduates",
    "Renters vs. Owners":                   "Renters vs. Owners",
    "Longer Summers":                       "Longer Summers",
    "Poor Air Quality":                     "Poor Air Quality",
    "Longer Driving Commute Time":          "Longer Driving Commute Time",
    "Walking to Work":                      "Walking to Work",       # proxy for Walkability
    "Riding Bike to Work":                  "Riding Bike to Work",   # proxy for Bikeability
    "Limited Supermarket Access":           "Limited Supermarket Access",
    "Foreign Born Population":              "Foreign Born Population",
    "Per-capita Household Income":          "Per-capita Household Income",
    "Population Density":                   "Population Density",
    "Uninsured (All Ages)":                 "Uninsured, All Ages",
}

TARGET_METRIC = "Public Transportation Use"

# Year to use for the cross-sectional snapshot
# 2019 is the last "normal" pre-COVID year with near-complete coverage
SNAPSHOT_YEAR = 2019

print("Loading raw data …")
df_raw = pd.read_csv(RAW_PATH, low_memory=False)

def extract_metric(df, metric_label, year=SNAPSHOT_YEAR):
    """Return city → value Series for a given metric/year (All race, Both sex)."""
    mask = (
        (df["metric_item_label"] == metric_label) &
        (df["date_label"] == year) &
        (df["strata_race_label"] == "All") &
        (df["strata_sex_label"] == "Both") &
        (df["geo_label_citystate"] != "U.S. Total")
    )
    sub = df.loc[mask, ["geo_label_citystate", "value"]].dropna()
    # If a city appears more than once (shouldn't happen for All/Both), take mean
    return sub.groupby("geo_label_citystate")["value"].mean()

# Build feature matrix
frames = {}
for feat_name, metric_label in FEATURE_MAP.items():
    frames[feat_name] = extract_metric(df_raw, metric_label)

frames["Public Transportation Use"] = extract_metric(df_raw, TARGET_METRIC)

df_wide = pd.DataFrame(frames).dropna()   # keep only cities with full data
print(f"\nSnapshot year : {SNAPSHOT_YEAR}")
print(f"Cities retained (no missing): {len(df_wide)}")
print(f"Features: {len(FEATURE_MAP)}\n")
print(df_wide[["Public Transportation Use"]].sort_values("Public Transportation Use", ascending=False).to_string())

# =============================================================================
# 2. Feature Matrix & Target
# =============================================================================

FEATURE_COLS = list(FEATURE_MAP.keys())

X_raw = df_wide[FEATURE_COLS].values
y     = df_wide["Public Transportation Use"].values
cities = df_wide.index.tolist()

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# =============================================================================
# 3. Helper utilities
# =============================================================================

def loo_eval(model, X, y, model_name):
    """Leave-one-out CV → RMSE and R²."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    rmse = np.sqrt(mean_squared_error(y, preds))
    # LOO R² : 1 - SS_res / SS_tot
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_loo  = 1 - ss_res / ss_tot
    print(f"  {model_name:<35s}  LOO-RMSE = {rmse:6.3f}   LOO-R² = {r2_loo:6.3f}")
    return rmse, r2_loo, preds


def kfold_eval(model, X, y, k=5, model_name=""):
    """K-fold CV → mean ± std RMSE."""
    kf  = KFold(n_splits=k, shuffle=True, random_state=SEED)
    neg_mse = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse)
    r2_scores   = cross_val_score(model, X, y, cv=kf, scoring="r2")
    print(f"  {model_name:<35s}  {k}-fold RMSE = {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}   "
          f"R² = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
    return rmse_scores, r2_scores

# =============================================================================
# 4. MODEL A – Linear Regression (OLS, Ridge, Lasso)
# =============================================================================
print("\n" + "="*65)
print("MODEL A: Linear Regression")
print("="*65)

# --- 4a. OLS (full model on scaled features) ---
ols = LinearRegression()
ols.fit(X_scaled, y)
y_ols_full = ols.predict(X_scaled)
print(f"\n  OLS in-sample  R² = {r2_score(y, y_ols_full):.3f}")

ols_rmse, ols_r2, ols_preds = loo_eval(LinearRegression(), X_scaled, y, "OLS")

# --- 4b. Ridge (alpha tuned via LOO) ---
alphas = np.logspace(-2, 4, 40)
ridge_scores = []
for a in alphas:
    _, r2_a, _ = loo_eval(Ridge(alpha=a), X_scaled, y, f"Ridge α={a:.3f}")
    ridge_scores.append(r2_a)
best_ridge_alpha = alphas[np.argmax(ridge_scores)]
ridge = Ridge(alpha=best_ridge_alpha)
ridge_rmse, ridge_r2, ridge_preds = loo_eval(ridge, X_scaled, y, f"Ridge (α={best_ridge_alpha:.3f})")

# --- 4c. Lasso (alpha tuned via LOO) ---
lasso_scores = []
for a in alphas:
    _, r2_a, _ = loo_eval(Lasso(alpha=a, max_iter=10000), X_scaled, y, f"Lasso α={a:.3f}")
    lasso_scores.append(r2_a)
best_lasso_alpha = alphas[np.argmax(lasso_scores)]
lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
lasso_rmse, lasso_r2, lasso_preds = loo_eval(lasso, X_scaled, y, f"Lasso (α={best_lasso_alpha:.3f})")

print(f"\n  Best Ridge α : {best_ridge_alpha:.4f}")
print(f"  Best Lasso α : {best_lasso_alpha:.4f}")

# OLS coefficients
ridge.fit(X_scaled, y)
coef_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "OLS Coef":   ols.coef_,
    "Ridge Coef": ridge.coef_,
}).sort_values("OLS Coef", key=abs, ascending=False)
print("\n  Top-10 OLS coefficients (absolute value):")
print(coef_df.head(10).to_string(index=False))

# =============================================================================
# 5. MODEL B – K-Nearest Neighbors
# =============================================================================
print("\n" + "="*65)
print("MODEL B: K-Nearest Neighbors")
print("="*65)

k_range = range(1, min(16, len(y)-1))
knn_loo_rmse = []
for k in k_range:
    rmse_k, _, _ = loo_eval(KNeighborsRegressor(n_neighbors=k), X_scaled, y, f"KNN k={k}")
    knn_loo_rmse.append(rmse_k)

best_k = list(k_range)[np.argmin(knn_loo_rmse)]
best_knn_rmse = min(knn_loo_rmse)
print(f"\n  Optimal k = {best_k}  (LOO-RMSE = {best_knn_rmse:.3f})")

knn = KNeighborsRegressor(n_neighbors=best_k)
knn_rmse, knn_r2, knn_preds = loo_eval(knn, X_scaled, y, f"KNN (k={best_k})")

# =============================================================================
# 6. MODEL C – Random Forest
# =============================================================================
print("\n" + "="*65)
print("MODEL C: Random Forest")
print("="*65)

# Coarse grid search via 5-fold CV (LOO too slow for RF)
param_grid = {
    "n_estimators": [100, 300],
    "max_depth":    [None, 3, 5],
    "min_samples_leaf": [1, 2, 3],
}
rf_base = RandomForestRegressor(random_state=SEED)
kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
gs  = GridSearchCV(rf_base, param_grid, cv=kf5,
                   scoring="neg_mean_squared_error", n_jobs=-1)
gs.fit(X_scaled, y)
best_params = gs.best_params_
print(f"\n  Best RF params: {best_params}")

rf = RandomForestRegressor(**best_params, random_state=SEED)
rf_rmse, rf_r2, rf_preds = loo_eval(rf, X_scaled, y, "Random Forest (tuned)")

# Feature importance from full fit
rf.fit(X_scaled, y)
imp_df = pd.DataFrame({
    "Feature":    FEATURE_COLS,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
print("\n  Top-10 RF feature importances:")
print(imp_df.head(10).to_string(index=False))

# Permutation importance (more reliable)
perm = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=SEED)
perm_df = pd.DataFrame({
    "Feature":     FEATURE_COLS,
    "Perm Importance": perm.importances_mean
}).sort_values("Perm Importance", ascending=False)
print("\n  Top-10 RF permutation importances:")
print(perm_df.head(10).to_string(index=False))

# =============================================================================
# 7. Summary Table
# =============================================================================
print("\n" + "="*65)
print("SUMMARY: LOO CV Performance")
print("="*65)
summary = pd.DataFrame({
    "Model":     ["OLS", f"Ridge (α={best_ridge_alpha:.4f})", f"Lasso (α={best_lasso_alpha:.4f})",
                  f"KNN (k={best_k})", "Random Forest"],
    "LOO-RMSE":  [ols_rmse, ridge_rmse, lasso_rmse, knn_rmse, rf_rmse],
    "LOO-R²":    [ols_r2,   ridge_r2,   lasso_r2,   knn_r2,   rf_r2],
})
print(summary.to_string(index=False))

# =============================================================================
# 8. Plots
# =============================================================================

# Colour cities by region
REGION = {
    "New York City, NY": "Northeast", "Boston, MA": "Northeast",
    "Washington, DC": "Northeast", "Philadelphia, PA": "Northeast",
    "Baltimore, MD": "Northeast",
    "Chicago, IL": "Midwest", "Detroit, MI": "Midwest",
    "Columbus, OH": "Midwest", "Cleveland, OH": "Midwest",
    "Milwaukee, WI": "Midwest", "Minneapolis, MN": "Midwest",
    "Kansas City, MO": "Midwest", "Indianapolis, IN": "Midwest",
    "Los Angeles, CA": "West", "San Francisco, CA": "West",
    "Seattle, WA": "West", "Portland, OR": "West",
    "Oakland, CA": "West", "San Diego, CA": "West",
    "San Jose, CA": "West", "Long Beach, CA": "West",
    "Denver, CO": "West", "Las Vegas, NV": "West",
    "Phoenix, AZ": "West", "Tucson, AZ": "West",
    "Houston, TX": "South", "Dallas, TX": "South",
    "San Antonio, TX": "South", "Austin, TX": "South",
    "El Paso, TX": "South", "Nashville, TN": "South",
    "Memphis, TN": "South", "Louisville, KY": "South",
    "Charlotte, NC": "South", "Oklahoma City, OK": "South",
}
REGION_COLORS = {
    "Northeast": "#e63946",
    "Midwest":   "#457b9d",
    "West":      "#2a9d8f",
    "South":     "#e9c46a",
    "Other":     "#aaa",
}

def city_color(city):
    region = REGION.get(city, "Other")
    return REGION_COLORS[region]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Supervised Learning: Predicting Public Transit Use\nLinear Algebros — Miguel Cuevas", fontsize=14, fontweight="bold")

# --- 8a. Actual vs Predicted for each model ---
model_preds = {
    "OLS":            ols_preds,
    f"Ridge":         ridge_preds,
    f"Lasso":         lasso_preds,
    f"KNN (k={best_k})": knn_preds,
    "Random Forest":  rf_preds,
}

ax_pairs = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
for ax, (mname, preds) in zip(ax_pairs, model_preds.items()):
    colors = [city_color(c) for c in cities]
    ax.scatter(y, preds, c=colors, s=60, edgecolors="k", linewidths=0.4, zorder=3)
    for i, city in enumerate(cities):
        short = city.split(",")[0]
        ax.annotate(short, (y[i], preds[i]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points", color="#333")
    lo, hi = min(y.min(), preds.min())-1, max(y.max(), preds.max())+1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect fit")
    rmse_val = np.sqrt(mean_squared_error(y, preds))
    ss_res = np.sum((y - preds)**2); ss_tot = np.sum((y - y.mean())**2)
    r2_val = 1 - ss_res/ss_tot
    ax.set_xlabel("Actual % Public Transit Use", fontsize=9)
    ax.set_ylabel("LOO-Predicted %", fontsize=9)
    ax.set_title(f"{mname}\nLOO-RMSE={rmse_val:.2f}, R²={r2_loo:.3f}", fontsize=9)
    ax.grid(alpha=0.3)
    # region legend once
    if mname == "OLS":
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=r, edgecolor="k") for r, c in REGION_COLORS.items() if r != "Other"]
        ax.legend(handles=handles, fontsize=6, loc="upper left", framealpha=0.7)

# --- 8b. Model comparison bar chart ---
ax6 = axes[1,2]
model_names_short = ["OLS", "Ridge", "Lasso", f"KNN\n(k={best_k})", "RF"]
rmse_vals = [ols_rmse, ridge_rmse, lasso_rmse, knn_rmse, rf_rmse]
r2_vals   = [ols_r2,   ridge_r2,   lasso_r2,   knn_r2,   rf_r2]
x_pos = np.arange(len(model_names_short))
bars = ax6.bar(x_pos, rmse_vals, color=["#4c72b0","#55a868","#c44e52","#8172b2","#ccb974"],
               edgecolor="k", linewidth=0.6, width=0.55)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(model_names_short, fontsize=9)
ax6.set_ylabel("LOO-RMSE (% commuters)", fontsize=9)
ax6.set_title("Model Comparison – LOO-RMSE\n(lower is better)", fontsize=9)
ax6.grid(axis="y", alpha=0.3)
for bar, r2v in zip(bars, r2_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"R²={r2v:.2f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/miguel_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: miguel_actual_vs_predicted.png")

# --- 8c. Feature Importance (RF & OLS coefficients) ---
fig, axes2 = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Feature Importance & Coefficients", fontsize=13, fontweight="bold")

# RF impurity importance
ax = axes2[0]
top_rf = imp_df.head(15)
colors_rf = ["#2a9d8f" if v > 0 else "#e63946" for v in top_rf["Importance"]]
ax.barh(top_rf["Feature"][::-1], top_rf["Importance"][::-1], color=colors_rf[::-1], edgecolor="k", linewidth=0.4)
ax.set_xlabel("Mean Decrease in Impurity", fontsize=9)
ax.set_title("Random Forest\nFeature Importance (top 15)", fontsize=10)
ax.grid(axis="x", alpha=0.3)

# RF permutation importance
ax = axes2[1]
top_perm = perm_df.head(15)
colors_pm = ["#457b9d" if v > 0 else "#e63946" for v in top_perm["Perm Importance"]]
ax.barh(top_perm["Feature"][::-1], top_perm["Perm Importance"][::-1], color=colors_pm[::-1], edgecolor="k", linewidth=0.4)
ax.set_xlabel("Mean Decrease in R²", fontsize=9)
ax.set_title("Random Forest\nPermutation Importance (top 15)", fontsize=10)
ax.grid(axis="x", alpha=0.3)

# OLS coefficients
ax = axes2[2]
ols_coef = pd.DataFrame({"Feature": FEATURE_COLS, "Coef": ols.coef_}).sort_values("Coef")
colors_ols = ["#e63946" if v < 0 else "#2a9d8f" for v in ols_coef["Coef"]]
ax.barh(ols_coef["Feature"], ols_coef["Coef"], color=colors_ols, edgecolor="k", linewidth=0.4)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coefficient (standardized features)", fontsize=9)
ax.set_title("OLS Coefficients\n(standardized)", fontsize=10)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/miguel_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: miguel_feature_importance.png")

# --- 8d. KNN: k vs RMSE ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), knn_loo_rmse, "o-", color="#457b9d", lw=2, ms=6)
ax.axvline(best_k, color="#e63946", lw=1.5, linestyle="--", label=f"Best k={best_k}")
ax.set_xlabel("Number of Neighbors (k)", fontsize=10)
ax.set_ylabel("LOO-RMSE", fontsize=10)
ax.set_title("KNN: Choosing k via Leave-One-Out CV", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/miguel_knn_k_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: miguel_knn_k_selection.png")

# --- 8e. Ridge/Lasso alpha tuning ---
fig, axes3 = plt.subplots(1, 2, figsize=(12, 4))
for ax, scores, best_a, label, color in [
    (axes3[0], ridge_scores, best_ridge_alpha, "Ridge", "#55a868"),
    (axes3[1], lasso_scores, best_lasso_alpha, "Lasso", "#c44e52"),
]:
    ax.semilogx(alphas, scores, "o-", color=color, lw=2, ms=4)
    ax.axvline(best_a, color="k", lw=1.5, linestyle="--", label=f"Best α={best_a:.4f}")
    ax.set_xlabel("α (regularization strength)", fontsize=10)
    ax.set_ylabel("LOO-R²", fontsize=10)
    ax.set_title(f"{label}: Choosing α via LOO CV", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/miguel_regularization_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: miguel_regularization_tuning.png")

# --- 8f. Correlation heatmap of features vs target ---
corr_data = df_wide[FEATURE_COLS + ["Public Transportation Use"]].corr()
target_corr = corr_data["Public Transportation Use"].drop("Public Transportation Use").sort_values()
fig, ax = plt.subplots(figsize=(8, 8))
colors_hm = ["#e63946" if v < 0 else "#2a9d8f" for v in target_corr]
ax.barh(target_corr.index, target_corr.values, color=colors_hm, edgecolor="k", linewidth=0.4)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson Correlation with Public Transit Use", fontsize=10)
ax.set_title("Feature Correlations with Target Variable\n(Public Transportation Use %)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/miguel_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: miguel_feature_correlations.png")

# --- 8g. Save cleaned wide-format data for teammates ---
df_wide.to_csv("/mnt/user-data/outputs/cities_features_2019.csv")
print("Saved: cities_features_2019.csv  (pivot table for teammates)")

print("\n✓ All outputs written to /mnt/user-data/outputs/")
print("\nDone.")
