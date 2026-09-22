"""
01_load_clean.py
Reads the raw FreshFork survey, recodes the brand statements to numeric,
defines the young families audience, and saves a clean dataset.
Every downstream script reads data/clean/responses.parquet, never the raw file.
"""

import pandas as pd
import numpy as np

PATH = "data/raw/freshfork_survey.xlsx"

demo  = pd.read_excel(PATH, sheet_name="Demographics")
brand = pd.read_excel(PATH, sheet_name="Brand Statements")
codes = pd.read_excel(PATH, sheet_name="Codemap")

# Verify the join before trusting it
print("Demographics:", demo.shape, "| Brand:", brand.shape)
print("Duplicate IDs:", demo.respid.duplicated().sum(), brand.respid.duplicated().sum())
print("IDs match:", set(demo.respid) == set(brand.respid))

# Statement columns store scale endpoints as text, so pandas reads them as
# strings. Recode to 1-10. "Don't know" becomes missing rather than a midpoint,
# because no opinion is not the same as a neutral opinion.
scale_cols = [c for c in brand.columns if c != "respid"]

def to_numeric(v):
    if v == "1 Completely disagree": return 1
    if v == "10 Completely agree":   return 10
    if v == "Don't know":            return np.nan
    return v

B = brand.copy()
for c in scale_cols:
    B[c] = B[c].map(to_numeric).astype(float)

print(B[scale_cols].describe().T[["count", "mean", "min", "max"]].head())

# Young families = children under 10, which is bands D8b_1 to D8b_4.
# D8b_5 (10-12) and D8b_6 (13-15) are excluded.
young_kid_cols = ["D8b_1", "D8b_2", "D8b_3", "D8b_4"]
demo["young_family"] = (demo[young_kid_cols] == "Selected").any(axis=1)
print(demo.young_family.value_counts())

# D8 records the child count and caps its top category at "4+", with a blank
# for no children under 16. Cross-tabbing D8 against the number of age bands
# ticked is a sanity check: nobody should tick more distinct bands than they
# have children, and counts below the diagonal are expected because two
# children can fall in the same band.
kid_bands = ["D8b_1","D8b_2","D8b_3","D8b_4","D8b_5","D8b_6"]
print(pd.crosstab(demo.D8, (demo[kid_bands] == "Selected").sum(axis=1)))

# age_band is delivered ready to use; keep a copy under a clear name.
demo["age_band"] = demo.dD2

# D8 caps at "4+", so the column mixes numbers and text. Stored as text since
# it is a category, not a quantity.
demo["D8"] = demo["D8"].astype("string")

df = demo.merge(B, on="respid")
df.to_parquet("data/clean/responses.parquet")
print("Saved:", df.shape)
