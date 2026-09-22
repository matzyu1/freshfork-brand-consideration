"""
04_halo.py
Can the 34 brand statements be ranked individually against consideration, or
does shared variance make that misleading?

1. Correlation of each statement with consideration
2. How much the statements correlate with each other
3. PCA, to size the shared dimension underneath them
4. Plain OLS regression, to show what happens if you ignore all this
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

PATH = "data/raw/freshfork_survey.xlsx"
df = pd.read_parquet("data/clean/responses.parquet")
codes = pd.read_excel(PATH, sheet_name="Codemap")

# zip pairs each variable code with its question text; dict makes it lookupable,
# so labels["B1_7"] returns the full statement.
labels = dict(zip(codes.Variable.astype(str), codes.Question.astype(str)))

# For every column name in df, keep it if it starts with B1.
scale_cols = [c for c in df.columns if c.startswith("B1")]

# Same idea, minus the outcome (B1c_33) and intent (B1c_34). Leaves 32.
predictors = [c for c in scale_cols if c not in ("B1c_33", "B1c_34")]
print("Predictors:", len(predictors))

# --- 1. Correlation with consideration -------------------------------------
# df[predictors] selects the 32 columns of interest.
# .corrwith(df.B1c_33) correlates each of those 32 columns against consideration.
corr = pd.DataFrame({"r": df[predictors].corrwith(df.B1c_33).round(3)})
corr["statement"] = [labels.get(i, "") for i in corr.index]
corr = corr.sort_values("r", ascending=False)

print("\nCorrelation with consideration, strongest")
print(corr.head(10).to_string())
print("\nWeakest")
print(corr.tail(5).to_string())

# --- 2. How much the statements overlap ------------------------------------
# Regression needs complete rows, so this is the population everything below uses.
X = df[predictors].dropna()
print("\nComplete cases:", len(X))

# Correlation matrix of the 32 statements, then average the upper triangle
# (k=1 skips the diagonal of 1.0s) to size how much they move together.
R = X.corr()
upper = R.values[np.triu_indices(len(predictors), k=1)]
print("Average correlation between statements: %.3f" % upper.mean())
print("Range: %.2f to %.2f" % (upper.min(), upper.max()))

# --- 3. PCA ----------------------------------------------------------------
# Standardise: every column mean 0, spread 1, so none dominates through scale.
Xs = (X - X.mean()) / X.std()
pca = PCA().fit(Xs)

# explained_variance_ratio_ gives the shared variance of each component. The
# first is the size of the single dimension sitting under all 32 statements.
var = pca.explained_variance_ratio_ * 100
print("\nVariance explained by first component: %.1f%%" % var[0])
print("First five components:", var[:5].round(1))

# --- 4. OLS regression -----------------------------------------------------
y = df.loc[X.index, "B1c_33"]
ok = y.notna()                 # drop rows where consideration itself is "Don't know"
Xr = Xs[ok]
yr = y[ok]
ys = (yr - yr.mean()) / yr.std()     # standardise the outcome too

# Column of ones gives the model an intercept; least squares finds the
# coefficients that minimise total squared error. [0] takes the coefficients,
# [1:] drops the intercept.
Xd = np.column_stack([np.ones(len(Xr)), Xr])
beta = np.linalg.lstsq(Xd, ys, rcond=None)[0][1:]

ols = pd.DataFrame({"beta": beta.round(3)}, index=predictors)
ols["statement"] = [labels.get(i, "") for i in ols.index]
ols = ols.sort_values("beta")

print("\nOLS standardised coefficients, most negative")
print(ols.head(8).to_string())
print("\nNegative coefficients: %d of %d" % ((ols.beta < 0).sum(), len(ols)))

# VIF measures how much one predictor overlaps with all the others: the
# diagonal of the inverse correlation matrix. The standard "> 10 is a problem"
# rule would pass this model as sound, which is exactly the trap.
vif = pd.Series(np.diag(np.linalg.inv(R.values)), index=predictors)
print("\nVIF: max %.1f, median %.1f, above 10: %d"
      % (vif.max(), vif.median(), (vif > 10).sum()))

# Each person's own average across the 32 statements is the halo itself.
halo = Xr.mean(axis=1)
print("\nHalo (person's own average) vs consideration: r=%.2f" % halo.corr(ys))

# Subtracting the halo isolates what each statement adds beyond general goodwill.
# Product/experience items survive; company and ethics items collapse or reverse.
print("\nRaw vs relative-to-own-average correlation")
for v in ["B1_7", "B1_9", "B1c_30", "B1b_14", "B1b_12"]:
    raw = Xr[v].corr(ys)
    rel = (Xr[v] - halo).corr(ys)
    print("%-8s raw %+.2f | relative %+.2f | %s" % (v, raw, rel, labels.get(v, "")[:45]))
