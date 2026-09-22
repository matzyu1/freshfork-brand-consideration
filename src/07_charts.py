"""
07_charts.py
Builds the six charts for the presentation. Exports PNG to output/charts/.

Reuses analysis already run and verified in earlier scripts. Nothing new is
calculated here; every number should match what 04, 05 and 06 printed.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend: this script only writes PNGs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.decomposition import FactorAnalysis


# Standard loading block, same as every script since 02_audit.py. Reads the
# clean parquet written by 01_load_clean.py, never the raw Excel, except for
# the Codemap which only exists there.
PATH = "data/raw/freshfork_survey.xlsx"
df = pd.read_parquet("data/clean/responses.parquet")
codes = pd.read_excel(PATH, sheet_name="Codemap")
labels = dict(zip(codes.Variable.astype(str), codes.Question.astype(str)))

scale_cols = [c for c in df.columns if c.startswith("B1")]
predictors = [c for c in scale_cols if c not in ("B1c_33", "B1c_34")]
FACTORS = ["F1", "F2", "F3", "F4"]

# Names chosen by reading the statements in each factor group printed by
# 05_drivers.py. Not machine labels.
FACTOR_NAMES = {
    "F1": "Product & experience",
    "F2": "Company & ethics",
    "F3": "Fame & familiarity",
    "F4": "Service",
}
FAME = "F3"          # the fame & familiarity factor, highlighted in chart 6

ACCENT = "#1f4e79"
GREY   = "#9a9a9a"
RED    = "#c0392b"
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

OUT = "output/charts/"

# --- Shared data prep ------------------------------------------------------
# All six lines below are lifted from 04_halo.py. Charts 2, 4 and 5 need them,
# so they sit here once rather than being repeated per chart.
X  = df[predictors].dropna()
Xs = (X - X.mean()) / X.std()

y  = df.loc[X.index, "B1c_33"]
ok = y.notna()

Xr, yr = Xs[ok], y[ok]
ys = (yr - yr.mean()) / yr.std()

# Each person's own average across all 32 statements: the halo itself.
halo = Xr.mean(axis=1)

print("Rows:", len(Xr))


# ===========================================================================
# Chart 1: consideration by age
# ===========================================================================
age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
by_age = df.groupby("age_band").B1c_33.mean().reindex(age_order)

plt.figure(figsize=(9, 5))
sns.barplot(x=by_age.index, y=by_age.values, color=ACCENT)
plt.title("Consideration by age")
plt.xlabel("Age band")
plt.ylabel("Mean consideration (1-10)")
plt.ylim(0, 10)
plt.tight_layout()
plt.savefig(OUT + "01_consideration_by_age.png", dpi=300, bbox_inches="tight")
plt.close()


# ===========================================================================
# Chart 2: the halo effect made visible
# ===========================================================================
# Same calculation as the final block of 04_halo.py, which printed these five
# statements as text. Here it becomes a slope chart.
SHORT = {
    "B1_7":   "Ethical and responsible",
    "B1_9":   "Contributes to my community",
    "B1c_30": "Has a good reputation",
    "B1b_14": "Great tasting food and drink",
    "B1b_12": "A place I enjoy visiting",
}

rows = []
for v in SHORT:
    rows.append({
        "statement": SHORT[v],
        "raw": Xr[v].corr(ys),
        "relative": (Xr[v] - halo).corr(ys),
    })
flip = pd.DataFrame(rows)
print(flip.round(2).to_string())

def tier_colour(rel):
    # Survives the halo removal (blue), vanishes to nothing (grey), or reverses
    # (red). Thresholds are a judgement call, not a standard.
    if rel > 0.05:
        return ACCENT
    if rel > -0.05:
        return GREY
    return RED

plt.figure(figsize=(9, 6))
for _, r in flip.iterrows():
    plt.plot([0, 1], [r["raw"], r["relative"]], "-o",
             color=tier_colour(r["relative"]), linewidth=2)

# Declutter the right-hand labels: keep them in value order but force a minimum
# vertical gap so they never overlap when two lines land close together.
order = flip.sort_values("relative", ascending=False).reset_index(drop=True)
min_gap = 0.052
label_y = order["relative"].tolist()
for i in range(1, len(label_y)):
    if label_y[i] > label_y[i - 1] - min_gap:
        label_y[i] = label_y[i - 1] - min_gap
for i, row in order.iterrows():
    plt.text(1.04, label_y[i], row["statement"], va="center", fontsize=11,
             color=tier_colour(row["relative"]))

plt.axhline(0, color="#555555", linewidth=1.2, linestyle="--")
plt.xticks([0, 1], ["Raw correlation", "After removing\ngeneral goodwill"])
plt.xlim(-0.1, 1.9)
plt.ylabel("Correlation with consideration")
plt.title("Correlation with consideration, raw and adjusted")
plt.tight_layout()
plt.savefig(OUT + "02_halo_flip.png", dpi=300, bbox_inches="tight")
plt.close()


# ===========================================================================
# Factor model
# ===========================================================================
# Identical to the top of 05_drivers.py: same data, same settings, same seed,
# so these are the same four factors. Charts 3, 4 and 5 all build on it.
fa = FactorAnalysis(n_components=4, rotation="varimax", random_state=0).fit(Xs)
for i in range(len(FACTORS)):
    if fa.components_[i].mean() < 0:
        fa.components_[i] *= -1

loadings = pd.DataFrame(fa.components_.T, index=predictors, columns=FACTORS)
loadings["belongs_to"] = loadings[FACTORS].abs().idxmax(axis=1)


# ===========================================================================
# Chart 3: factor structure (appendix)
# ===========================================================================
# Statement counts per factor. Appendix only: the count is not a finding, and a
# long bar could be misread as meaning that factor matters most. Importance
# comes from chart 4.
counts = loadings.belongs_to.value_counts().reindex(FACTORS)
names = [FACTOR_NAMES[f] for f in FACTORS]

plt.figure(figsize=(9, 4))
sns.barplot(x=counts.values, y=names, color=ACCENT)
plt.title("Factor structure: how the 32 statements group")
plt.xlabel("Number of statements")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUT + "03_factor_structure.png", dpi=300, bbox_inches="tight")
plt.close()


# ===========================================================================
# Chart 4: importance vs performance
# ===========================================================================
scores = pd.DataFrame(fa.transform(Xs), index=X.index, columns=FACTORS)
Xf = scores[ok].values

Xd = np.column_stack([np.ones(len(Xf)), Xf])
result = np.linalg.lstsq(Xd, ys, rcond=None)[0]
betas = result[1:]
shares = (betas ** 2) / (betas ** 2).sum() * 100

perf = [df[loadings[loadings.belongs_to == f].index].mean().mean()
        for f in FACTORS]

iv = pd.DataFrame({
    "factor": [FACTOR_NAMES[f] for f in FACTORS],
    "importance": shares,
    "performance": perf,
})
print(iv.round(2).to_string())

iv_sorted = iv.sort_values("importance", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

sns.barplot(x="importance", y="factor", data=iv_sorted, color=ACCENT, ax=axes[0])
axes[0].set_title("Importance")
axes[0].set_xlabel("% of explained variance")
axes[0].set_ylabel("")
axes[0].set_xlim(0, 65)
for i, v in enumerate(iv_sorted["importance"]):
    axes[0].text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=11)

# Grey rather than blue because performance is a different measure on a
# different scale, not a comparable bar.
sns.barplot(x="performance", y="factor", data=iv_sorted, color=GREY, ax=axes[1])
axes[1].set_title("Current performance")
axes[1].set_xlabel("Mean score (1-10)")
axes[1].set_ylabel("")
axes[1].set_xlim(0, 10)
for i, v in enumerate(iv_sorted["performance"]):
    axes[1].text(v + 0.2, i, f"{v:.2f}", va="center", fontsize=11)

fig.suptitle("Importance vs current performance, by factor", fontsize=15)
plt.tight_layout()
plt.savefig(OUT + "04_importance_vs_performance.png", dpi=300, bbox_inches="tight")
plt.close()


# ===========================================================================
# Chart 5: factor importance, young families vs everyone else
# ===========================================================================
# Same comparison as block 1 of 06_subgroup.py. One factor structure fitted on
# the whole sample (above), then the regression run separately per group, so
# both are measured on the same four dimensions.
def factor_regression(data_scores, consideration):
    """Returns each factor's share of explained variance."""
    Xg = data_scores.values
    yg = ((consideration - consideration.mean()) / consideration.std()).values
    Xd = np.column_stack([np.ones(len(Xg)), Xg])
    betas = np.linalg.lstsq(Xd, yg, rcond=None)[0][1:]
    return (betas ** 2) / (betas ** 2).sum() * 100


sc   = scores[ok]
cons = y[ok]
yf_flag = df.loc[sc.index, "young_family"]

yf_shares  = factor_regression(sc[yf_flag],  cons[yf_flag])
oth_shares = factor_regression(sc[~yf_flag], cons[~yf_flag])

comp = pd.DataFrame({
    "factor": [FACTOR_NAMES[f] for f in FACTORS],
    "Young families": yf_shares,
    "Everyone else": oth_shares,
})
print(comp.round(1).to_string())

long = comp.melt(id_vars="factor", var_name="group", value_name="importance")

plt.figure(figsize=(10, 5))
sns.barplot(data=long, x="importance", y="factor", hue="group",
            palette=[ACCENT, GREY])
plt.title("Factor importance: young families vs everyone else")
plt.xlabel("% of explained variance")
plt.ylabel("")
plt.legend(title="")
plt.tight_layout()
plt.savefig(OUT + "05_young_family_importance.png", dpi=300, bbox_inches="tight")
plt.close()


# ===========================================================================
# Chart 6: where young families differ, statement by statement
# ===========================================================================
# Same as block 3 of 06_subgroup.py. Single .mean() per group this time, giving
# one score per statement rather than per factor, then the difference.
gaps = pd.DataFrame({
    "young_family":  df[df.young_family][predictors].mean(),
    "everyone_else": df[~df.young_family][predictors].mean(),
})
gaps["gap"] = gaps.young_family - gaps.everyone_else
gaps["factor"] = loadings.belongs_to

# A couple of statements truncate mid-word at 40 characters, so they get
# hand-written labels; the rest fall back to the wording with the brand removed.
MANUAL = {
    "B1_9":   "contributes to my community",
    "B1c_25": "is a company I feel good about",
}
gaps["statement"] = [
    MANUAL.get(i, labels.get(i, "").replace("FreshFork ", "")[:40])
    for i in gaps.index
]

gaps = gaps.sort_values("gap", ascending=False)
ends = pd.concat([gaps.head(6), gaps.tail(6)])

plt.figure(figsize=(10, 7))
# Colour by factor so the reader can see the two smallest gaps are both
# fame & familiarity statements.
colours = [ACCENT if f == FAME else GREY for f in ends.factor]
plt.barh(range(len(ends)), ends.gap, color=colours)
plt.yticks(range(len(ends)), ends.statement, fontsize=11)
plt.gca().invert_yaxis()

plt.legend(handles=[
    Patch(color=ACCENT, label="Fame & familiarity"),
    Patch(color=GREY, label="Other factors"),
], loc="lower right")
plt.xlabel("Score difference (young families minus everyone else)")
plt.title("Where young families rate FreshFork differently")
plt.tight_layout()
plt.savefig(OUT + "06_statement_gaps.png", dpi=300, bbox_inches="tight")
plt.close()

print("Charts written to", OUT)
