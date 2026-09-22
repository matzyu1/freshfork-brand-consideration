"""
03_eda.py
Describes brand consideration (B1c_33), how it varies by demographic, and
where FreshFork scores strongly and weakly across the 34 brand statements.
"""

import pandas as pd
import numpy as np
from scipy import stats

PATH = "data/raw/freshfork_survey.xlsx"
df = pd.read_parquet("data/clean/responses.parquet")
codes = pd.read_excel(PATH, sheet_name="Codemap")
labels = dict(zip(codes.Variable.astype(str), codes.Question.astype(str)))

scale_cols = [c for c in df.columns if c.startswith("B1")]
print("Loaded:", df.shape, "| statements:", len(scale_cols))

# --- Consideration overall -------------------------------------------------
print("\n" + labels["B1c_33"])
print(df.B1c_33.describe().round(2).to_string())
print("Top 3 box (8-10): %.1f%%" % ((df.B1c_33 >= 8).mean() * 100))
print("Bottom 3 box (1-3): %.1f%%" % ((df.B1c_33 <= 3).mean() * 100))

# --- Consideration by demographic ------------------------------------------
for col in ["age_band", "D1", "young_family"]:
    print(f"\nConsideration by {col}")
    print(df.groupby(col).B1c_33.agg(["mean", "count"]).round(2).to_string())

# --- Is the young families lift just an age effect? ------------------------
# The two groups are entangled: young families cluster in the 25-44 bands and
# almost vanish over 55, and consideration falls with age, so the raw lift
# could be an age effect in disguise. The 35-44 band is where both groups are
# well represented, giving the cleanest like-for-like comparison, with 35-54
# as a wider secondary check.
print("\nYoung families by age band")
print(pd.crosstab(df.age_band, df.young_family).to_string())

for name, subset in [("35-44", ["35-44"]), ("35-54", ["35-44", "45-54"])]:
    sub = df[df.age_band.isin(subset)]
    print(f"\nConsideration within {name}")
    print(sub.groupby("young_family").B1c_33.agg(["mean", "count"]).round(2).to_string())

band = df[df.age_band == "35-44"]
t, p = stats.ttest_ind(band[band.young_family].B1c_33.dropna(),
                       band[~band.young_family].B1c_33.dropna())
print("\n35-44 young families vs others: t=%.2f, p=%.4f" % (t, p))

# --- Where FreshFork scores strongly and weakly ----------------------------
means = pd.DataFrame({"mean": df[scale_cols].mean().round(2)})
means["statement"] = [labels.get(i, "") for i in means.index]
means = means.sort_values("mean", ascending=False)

print("\nHighest scoring statements")
print(means.head(8).to_string())
print("\nLowest scoring statements")
print(means.tail(8).to_string())

means.to_csv("output/tables/statement_means.csv")
