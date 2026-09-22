"""
02_audit.py
Data quality audit: "Don't know" patterns, straightlining, sample composition.
Reads the clean dataset and the raw statements sheet, which is needed because
"Don't know" has already become missing in the clean file.
"""

import pandas as pd
import numpy as np

PATH = "data/raw/freshfork_survey.xlsx"

df    = pd.read_parquet("data/clean/responses.parquet")
brand = pd.read_excel(PATH, sheet_name="Brand Statements")
codes = pd.read_excel(PATH, sheet_name="Codemap")

labels = dict(zip(codes.Variable.astype(str), codes.Question.astype(str)))
scale_cols = [c for c in brand.columns if c != "respid"]

print("Loaded:", df.shape)

# "Don't know" rates. Comparing the top and bottom of this list is the finding:
# people hold views on eating at FreshFork but not on FreshFork as a company.
dk = pd.DataFrame({"dk_pct": ((brand[scale_cols] == "Don't know").mean() * 100).round(1)})
dk["statement"] = [labels.get(i, "") for i in dk.index]

print("\nHighest 'Don't know' rates")
print(dk.sort_values("dk_pct", ascending=False).head(10).to_string())
print("\nLowest 'Don't know' rates")
print(dk.sort_values("dk_pct").head(6).to_string())

# Straightliners: identical answer to all 34 statements. Breaking them down by
# the value chosen matters. A flat 5 is plausible indifference on a 10-point
# scale; a flat 10 across statements many people can't answer is not.
# Kept in the main analysis, tested as a sensitivity check in 05_drivers.
straight = df[scale_cols].std(axis=1) == 0
print("\nStraightliners:", straight.sum(), f"({straight.mean()*100:.1f}%)")
print(df[straight][scale_cols].mean(axis=1).value_counts().to_string())

# Sample skews female and toward 35-44. No weighting variables are provided,
# so results describe this sample rather than the UK population.
print("\nGender")
print(df.D1.value_counts(normalize=True).round(3).to_string())
print("\nAge band")
print(df.age_band.value_counts(normalize=True).round(3).to_string())
print("Mean age:", df.D2.mean().round(1))
