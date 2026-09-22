"""
06_subgroup.py
Do young families differ from the rest of the sample?

Uses a single factor structure fitted on the whole sample, so both groups are
measured on the same four dimensions. Fitting separate factor models per group
could produce different dimensions in each, making comparison meaningless.

Three separate questions:
  1. Do the groups weight the four factors differently?
  2. Do the groups rate FreshFork differently on each factor?
  3. Which individual statements show the biggest gap?
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis

PATH = "data/raw/freshfork_survey.xlsx"
df = pd.read_parquet("data/clean/responses.parquet")
codes = pd.read_excel(PATH, sheet_name="Codemap")
labels = dict(zip(codes.Variable.astype(str), codes.Question.astype(str)))

scale_cols = [c for c in df.columns if c.startswith("B1")]
predictors = [c for c in scale_cols if c not in ("B1c_33", "B1c_34")]
FACTORS = ["F1", "F2", "F3", "F4"]

# Filter and standardise, then fit the shared four-factor structure.
X = df[predictors].dropna()
Xs = (X - X.mean()) / X.std()

fa = FactorAnalysis(n_components=4, rotation="varimax", random_state=0).fit(Xs)
for i in range(len(FACTORS)):
    if fa.components_[i].mean() < 0:
        fa.components_[i] *= -1

loadings = pd.DataFrame(fa.components_.T, index=predictors, columns=FACTORS)
loadings["belongs_to"] = loadings[FACTORS].abs().idxmax(axis=1)

scores = pd.DataFrame(fa.transform(Xs), index=X.index, columns=FACTORS)
scores["young_family"] = df.loc[X.index, "young_family"]
scores["consideration"] = df.loc[X.index, "B1c_33"]
scores = scores[scores.consideration.notna()]


# ===========================================================================
# 1. Do the groups weight the four factors differently?
# ===========================================================================

def factor_regression(data):
    Xf = data[FACTORS].values
    y = data["consideration"]
    ys = ((y - y.mean()) / y.std()).values

    Xd = np.column_stack([np.ones(len(Xf)), Xf])
    result = np.linalg.lstsq(Xd, ys, rcond=None)[0]
    betas = result[1:]

    predicted = Xd @ result
    r2 = 1 - ((ys - predicted) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()

    # Squared betas as a share of the total, valid because the factors are
    # near-uncorrelated so their squared coefficients can be added up.
    shares = (betas ** 2) / (betas ** 2).sum() * 100
    return pd.Series(shares, index=FACTORS), r2, len(data)


groups = {
    "Total":         scores,
    "Young family":  scores[scores.young_family],
    "Everyone else": scores[~scores.young_family],
}

importance = {}
print("=" * 70)
print("1. FACTOR IMPORTANCE BY GROUP (% of explained variance)")
print("=" * 70)

for name, data in groups.items():
    shares, r2, n = factor_regression(data)
    importance[name] = shares
    print("%-15s R2=%.3f  n=%d" % (name, r2, n))

print()
print(pd.DataFrame(importance).round(1).to_string())


# ===========================================================================
# 2. Do the groups rate FreshFork differently on each factor?
# ===========================================================================

print("\n" + "=" * 70)
print("2. FACTOR PERFORMANCE BY GROUP (mean score out of 10)")
print("=" * 70)

perf = {}
for f in FACTORS:
    members = loadings[loadings.belongs_to == f].index
    perf[f] = {
        "Young family":  df[df.young_family][members].mean().mean(),
        "Everyone else": df[~df.young_family][members].mean().mean(),
    }

perf = pd.DataFrame(perf).T
perf["gap"] = perf["Young family"] - perf["Everyone else"]
print(perf.round(2).to_string())


# ===========================================================================
# 3. Which individual statements show the biggest gap?
# ===========================================================================

print("\n" + "=" * 70)
print("3. BIGGEST STATEMENT-LEVEL GAPS")
print("=" * 70)

gaps = pd.DataFrame({
    "young_family":  df[df.young_family][predictors].mean(),
    "everyone_else": df[~df.young_family][predictors].mean(),
})
gaps["gap"] = gaps.young_family - gaps.everyone_else
gaps["statement"] = [labels.get(i, "") for i in gaps.index]
gaps["factor"] = loadings.belongs_to
gaps = gaps.sort_values("gap", ascending=False)

print("\nYoung families rate these highest relative to others")
print(gaps.head(8).round(2).to_string())
print("\nSmallest gaps")
print(gaps.tail(5).round(2).to_string())

gaps.to_csv("output/tables/young_family_gaps.csv")
