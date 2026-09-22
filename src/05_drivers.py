"""
05_drivers.py
What drives consideration?

04_halo.py established that the 32 statements overlap too heavily to rank
individually: they correlate strongly on average, one dimension explains about
two thirds of their variance, and plain OLS on all 32 produces negative
coefficients on statements that are positive on their own.

Approach here: reduce the 32 statements to four underlying factors, then run
ordinary regression on those. Factors are near-uncorrelated by construction,
so the collinearity problem goes away and the coefficients behave.

Factors are numbered F1-F4 as the model returns them; the human labels
(product & experience, company & ethics, fame & familiarity, service) come
from reading each factor's statements below.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis

PATH = "data/raw/freshfork_survey.xlsx"
df = pd.read_parquet("data/clean/responses.parquet")
codes = pd.read_excel(PATH, sheet_name="Codemap")
labels = dict(zip(codes.Variable.astype(str), codes.Question.astype(str)))

scale_cols = [c for c in df.columns if c.startswith("B1")]

# B1c_33 (consideration) is the outcome. B1c_34 (intent to eat within a month)
# sits downstream of consideration and measures nearly the same thing, so both
# are held out of the predictor set. Leaves 32.
predictors = [c for c in scale_cols if c not in ("B1c_33", "B1c_34")]
FACTORS = ["F1", "F2", "F3", "F4"]


# ===========================================================================
# 1. Reduce 32 statements to four factors
# ===========================================================================

# Drop respondents who answered "Don't know" to any of the 32 statements, then
# standardise so no statement dominates purely by having a wider spread.
X = df[predictors].dropna()
Xs = (X - X.mean()) / X.std()

# n_components=4 extracts four factors; varimax rotation redistributes them so
# each statement loads strongly on one factor and weakly on the others;
# random_state fixes the result.
fa = FactorAnalysis(n_components=4, rotation="varimax", random_state=0).fit(Xs)

# Factor analysis is blind to sign, so a factor can come out inverted. Flipping
# any factor whose loadings average negative keeps "higher = better" without
# changing the maths.
for i in range(len(FACTORS)):
    if fa.components_[i].mean() < 0:
        fa.components_[i] = fa.components_[i] * -1

loadings = pd.DataFrame(fa.components_.T, index=predictors, columns=FACTORS)
loadings["statement"] = [labels.get(i, "") for i in loadings.index]
# Assign each statement to the factor it loads on most strongly.
loadings["belongs_to"] = loadings[FACTORS].abs().idxmax(axis=1)

print("=" * 70)
print("FACTOR STRUCTURE")
print("=" * 70)
for f in FACTORS:
    group = loadings[loadings.belongs_to == f].sort_values(f, ascending=False)
    print(f"\n--- {f} ({len(group)} statements) ---")
    print(group[[f, "statement"]].round(2).to_string())

loadings.to_csv("output/tables/factor_loadings.csv")


# ===========================================================================
# 2. Regress consideration on the four factors
# ===========================================================================

# transform() turns each person's 32 standardised answers into four factor scores.
scores = pd.DataFrame(fa.transform(Xs), index=X.index, columns=FACTORS)

y = df.loc[X.index, "B1c_33"]
ok = y.notna()
Xf = scores[ok]
yf = y[ok]
ys = (yf - yf.mean()) / yf.std()     # standardise the outcome

# Add an intercept column, then least squares. [0] takes the coefficients.
Xd = np.column_stack([np.ones(len(Xf)), Xf.values])
result = np.linalg.lstsq(Xd, ys, rcond=None)[0]
intercept, betas = result[0], result[1:]

predicted = Xd @ result
r2 = 1 - ((ys - predicted) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()

# Share of explained variance per factor. Squaring the standardised betas and
# normalising works here only because the four factors barely correlate.
factor_model = pd.DataFrame({"beta": betas.round(3)}, index=FACTORS)
factor_model["share_pct"] = ((factor_model.beta ** 2) /
                             (factor_model.beta ** 2).sum() * 100).round(1)

print("\n" + "=" * 70)
print("FOUR-FACTOR REGRESSION")
print("=" * 70)
print("R-squared: %.3f on n=%d" % (r2, len(Xf)))
print(factor_model.sort_values("beta", ascending=False).to_string())


# ===========================================================================
# 3. Current performance on each factor
# ===========================================================================

print("\n" + "=" * 70)
print("PERFORMANCE BY FACTOR")
print("=" * 70)

for f in FACTORS:
    members = loadings[loadings.belongs_to == f].index
    print("%s: mean score %.2f across %d statements"
          % (f, df[members].mean().mean(), len(members)))


# ===========================================================================
# 4. Sensitivity check: does the ranking hold without straightliners?
# ===========================================================================
# 02_audit.py found respondents who gave an identical answer to all 34
# statements. They were kept, because the group mixes plausible indifference
# (a flat 5) with likely disengagement (a flat 10) and the two cannot be
# cleanly separated. Rerunning the model without them tests that decision.

print("\n" + "=" * 70)
print("SENSITIVITY CHECK: EXCLUDING STRAIGHTLINERS")
print("=" * 70)

straight = df[scale_cols].std(axis=1) == 0
df_no_flat = df[~straight]
print("Excluded %d respondents, %d remain" % (straight.sum(), len(df_no_flat)))

# Refit the whole model on the reduced sample: same settings, same seed.
X2 = df_no_flat[predictors].dropna()
Xs2 = (X2 - X2.mean()) / X2.std()

fa2 = FactorAnalysis(n_components=4, rotation="varimax", random_state=0).fit(Xs2)
for i in range(len(FACTORS)):
    if fa2.components_[i].mean() < 0:
        fa2.components_[i] = fa2.components_[i] * -1

scores2 = pd.DataFrame(fa2.transform(Xs2), index=X2.index, columns=FACTORS)
y2 = df_no_flat.loc[X2.index, "B1c_33"]
ok2 = y2.notna()
Xf2 = scores2[ok2]
ys2 = (y2[ok2] - y2[ok2].mean()) / y2[ok2].std()

Xd2 = np.column_stack([np.ones(len(Xf2)), Xf2.values])
betas2 = np.linalg.lstsq(Xd2, ys2, rcond=None)[0][1:]
shares2 = (betas2 ** 2) / (betas2 ** 2).sum() * 100

# The factors are refitted, so in principle the labels could attach to different
# groupings; worth eyeballing the loadings if any share moves sharply.
compare = pd.DataFrame({
    "all respondents": factor_model["share_pct"],
    "excluding straightliners": pd.Series(shares2, index=FACTORS).round(1),
})
compare["difference"] = (compare["excluding straightliners"]
                         - compare["all respondents"]).round(1)
print(compare.to_string())
