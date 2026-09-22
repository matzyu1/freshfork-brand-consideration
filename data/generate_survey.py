"""
generate_survey.py
Generates a synthetic brand-tracking survey for FreshFork Grill, a fictional
UK quick-service restaurant, and writes it as a raw Excel workbook with the
same shape a real fieldwork export would have.

The data is fabricated, but it is not random. A latent-variable model gives it
the structure a real brand-perception survey has, so the analysis in src/ finds
genuine patterns rather than noise:

  * a dominant "halo" dimension (people who like FreshFork rate it well on
    everything), which is what makes naive driver ranking misleading;
  * a weaker four-factor structure underneath (product & experience, company &
    ethics, fame & familiarity, service);
  * consideration driven mainly by the *unique* part of product & experience,
    so corporate and ethics statements look important on their own but collapse
    once general goodwill is removed;
  * a young-families audience that is warmer across the board but wants the
    same things as everyone else.

It also injects the messiness a real export carries, so the cleaning and audit
steps in src/ have real work to do: scale endpoints stored as text, correlated
"Don't know" responses concentrated on corporate questions, and a small number
of straightliners.

Run once from the project root:  python data/generate_survey.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20240909)
N = 2377
BRAND = "FreshFork"

# ---------------------------------------------------------------------------
# Statement inventory: code, wording, factor, target mean, "Don't know" rate,
# and the young-families score lift. Factors:
#   P = product & experience, C = company & ethics, F = fame & familiarity,
#   S = service. The two outcome items (consideration, intent) sit outside the
#   factor model and are marked "Y".
# ---------------------------------------------------------------------------
# "Don't know" rates: high on citizenship claims people cannot judge (employer,
# community, ethics, environment), low on everyday experience they can. The
# citizenship items are drawn correlated (see below), so they go unanswered
# together rather than independently.
CITIZENSHIP = {"B1_6", "B1_7", "B1_8", "B1_9", "B1_10", "B1c_30"}

STATEMENTS = [
    # code,     factor, mean,  dk,    yf_lift, wording
    ("B1_1",    "C", 5.83, 0.010, 0.48, "{b} is a brand I love"),
    ("B1_2",    "C", 6.21, 0.010, 0.45, "{b} is a brand I trust"),
    ("B1_3",    "C", 6.25, 0.008, 0.44, "I have a positive opinion of {b}"),
    ("B1_4",    "P", 6.22, 0.010, 0.40, "I would be disappointed if {b} went out of business tomorrow"),
    ("B1_5",    "C", 6.07, 0.010, 0.42, "I would speak positively about {b} to friends/colleagues"),
    ("B1_6",    "C", 6.14, 0.045, 0.43, "{b} is a brand changing for the better"),
    ("B1_7",    "C", 5.79, 0.092, 0.40, "{b} is an ethical and responsible company"),
    ("B1_8",    "C", 6.29, 0.091, 0.42, "{b} is making changes to reduce its environmental impact"),
    ("B1_9",    "C", 5.65, 0.096, 0.55, "{b} makes a positive contribution to my community"),
    ("B1_10",   "C", 5.97, 0.162, 0.40, "{b} is a good employer"),
    ("B1_11",   "P", 6.65, 0.008, 0.27, "{b} has a good range of food and drinks"),
    ("B1b_12",  "P", 5.98, 0.008, 0.57, "{b} is a place I enjoy visiting"),
    ("B1b_13",  "P", 6.31, 0.010, 0.33, "{b} often brings out exciting new products"),
    ("B1b_14",  "P", 6.60, 0.011, 0.30, "{b} has great tasting food and drink"),
    ("B1b_15",  "P", 6.61, 0.010, 0.27, "{b} has great tasting burgers"),
    ("B1b_16",  "P", 6.35, 0.010, 0.28, "{b} has great tasting chicken"),
    ("B1b_17",  "P", 5.48, 0.010, 0.45, "{b} offers food I feel good about eating"),
    ("B1b_18",  "P", 6.24, 0.013, 0.35, "{b} is improving the nutritional content of its food"),
    ("B1b_19",  "P", 6.12, 0.010, 0.33, "{b} uses quality ingredients"),
    ("B1b_20",  "P", 6.31, 0.008, 0.34, "{b} has good quality food and drink"),
    ("B1b_21",  "P", 6.74, 0.008, 0.18, "{b} offers good value for money"),
    ("B1b_22",  "P", 5.96, 0.010, 0.65, "{b} makes my life easier"),
    ("B1c_23",  "S", 5.81, 0.010, 0.50, "{b} makes me feel like a valued customer"),
    ("B1c_24",  "P", 5.96, 0.010, 0.44, "{b} is a brand for someone like me"),
    ("B1c_25",  "P", 5.88, 0.012, 0.52, "{b} is a company I feel good about buying from"),
    ("B1c_26",  "F", 7.24, 0.010, 0.16, "{b} is a place for kids and families"),
    ("B1c_27",  "F", 6.52, 0.011, 0.30, "{b} is a brand I know a lot about"),
    ("B1c_28",  "F", 6.86, 0.010, 0.33, "{b} is a brand I see and hear a lot about"),
    ("B1c_29",  "F", 7.92, 0.008, 0.05, "{b} is an established brand"),
    ("B1c_30",  "C", 6.38, 0.050, 0.41, "{b} has a good reputation"),
    ("B1c_31",  "S", 6.41, 0.010, 0.27, "{b} has good customer service"),
    ("B1c_32",  "S", 6.48, 0.008, 0.28, "{b} has friendly staff"),
    ("B1c_33",  "Y", 6.40, 0.010, 0.78, "{b} is a restaurant I would consider eating at in the next month"),
    ("B1c_34",  "Y", 5.91, 0.015, 0.64, "{b} is a restaurant I intend to eat at in the next month"),
]

codes    = [s[0] for s in STATEMENTS]
factor   = {s[0]: s[1] for s in STATEMENTS}
mean_of  = {s[0]: s[2] for s in STATEMENTS}
dk_of    = {s[0]: s[3] for s in STATEMENTS}
yf_lift  = {s[0]: s[4] for s in STATEMENTS}
wording  = {s[0]: s[5].format(b=BRAND) for s in STATEMENTS}

predictors = [c for c in codes if factor[c] != "Y"]

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
BAND_P = [0.25, 0.18, 0.31, 0.12, 0.08, 0.06]
BAND_AGE = {"18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
            "45-54": (45, 54), "55-64": (55, 64), "65+": (65, 80)}

age_band = RNG.choice(BANDS, size=N, p=BAND_P)
age = np.array([RNG.integers(BAND_AGE[b][0], BAND_AGE[b][1] + 1) for b in age_band])
gender = RNG.choice(["Female", "Male"], size=N, p=[0.64, 0.36])

REGIONS = ["London", "South East", "South West", "East of England",
           "West Midlands", "East Midlands", "Yorkshire", "North West",
           "North East", "Scotland", "Wales", "Northern Ireland"]
region = RNG.choice(REGIONS, size=N)

# Young families: parents of a child under 10. Probability depends on age, so
# the group is entangled with age exactly as in real life (few over-55s, heavy
# in the 25-44 bands) -- which is what forces the age-controlled comparison.
YF_P = {"18-24": 0.12, "25-34": 0.88, "35-44": 0.61,
        "45-54": 0.15, "55-64": 0.02, "65+": 0.00}
young_family = RNG.random(N) < np.array([YF_P[b] for b in age_band])

# Child age bands (D8b_1 under 1 ... D8b_6 13-15) and child count D8, capped 4+.
kid_band_cols = [f"D8b_{i}" for i in range(1, 7)]
kid_bands = {c: np.array(["Not selected"] * N, dtype=object) for c in kid_band_cols}
n_children = np.zeros(N, dtype=int)

for i in range(N):
    if young_family[i]:
        k = RNG.integers(1, 4)                       # 1-3 young children
        chosen = RNG.choice([1, 2, 3, 4], size=k, replace=True)  # under-10 bands
        for band in chosen:
            kid_bands[f"D8b_{band}"][i] = "Selected"
        # some young families also have an older child
        if RNG.random() < 0.25:
            kid_bands[f"D8b_{RNG.choice([5, 6])}"][i] = "Selected"
        n_children[i] = min((k + (RNG.random() < 0.25)), 4)
    else:
        if RNG.random() < 0.24:                       # older-children household
            k = RNG.integers(1, 3)
            for band in RNG.choice([5, 6], size=k, replace=True):
                kid_bands[f"D8b_{band}"][i] = "Selected"
            n_children[i] = min(k, 4)
        # else: no children under 16

# D8 stores the child count as a label, capped at "4+", blank if none.
D8 = np.array(["" if n == 0 else ("4+" if n >= 4 else str(n)) for n in n_children],
              dtype=object)

# ---------------------------------------------------------------------------
# Latent brand-perception model
# ---------------------------------------------------------------------------
# g = general affinity (the halo). Age and gender shift it; young-family warmth
# is added later as explicit per-statement lifts so its shape can be controlled.
female = (gender == "Female").astype(float)
g = (-0.20 * (age - 39) / 12 + 0.05 * female
     + RNG.normal(0, 1.0, N))
g = (g - g.mean()) / g.std()

# Four factor latents: each is the halo plus a factor-specific part u_k. The
# small factor-specific weight keeps a single dimension dominant while leaving
# enough structure for factor analysis to recover four factors.
S_FACTOR = 0.62          # weight of the factor-specific component
u = {k: RNG.normal(0, 1, N) for k in ["P", "C", "F", "S"]}
f_latent = {k: g + S_FACTOR * u[k] for k in u}

NOISE = 0.60             # per-statement idiosyncratic noise
SLOPE = 1.55             # spreads latent onto the 1-10 scale

# Draw each statement's idiosyncratic noise up front, so consideration can
# borrow one of them (reputation) below.
stmt_noise = {c: RNG.normal(0, 1, N) for c in predictors}

raw = {}
for c in predictors:
    k = factor[c]
    z = f_latent[k] + NOISE * stmt_noise[c]
    score = mean_of[c] + SLOPE * z + yf_lift[c] * young_family
    raw[c] = score

# Consideration: general affinity, plus the *unique* part of product/experience
# (the dominant driver), a little service and a little fame. Company & ethics
# add nothing beyond the halo, which is what makes those statements collapse
# once each person's own average is removed.
cons_latent = (g
               + 0.52 * u["P"]
               + 0.27 * u["S"]
               + 0.27 * u["F"]
               + 0.30 * stmt_noise["B1c_30"]     # reputation carries a little
               + RNG.normal(0, 0.66, N))          # consideration-specific signal,
# so once the halo is removed it neither survives like product nor reverses like
# ethics -- it lands near zero, the classic "reputation is a lagging read-out of
# consideration, not a driver of it" pattern.
cons_latent = cons_latent / cons_latent.std()
raw["B1c_33"] = 6.30 + 2.70 * cons_latent + yf_lift["B1c_33"] * young_family
# Intent sits just downstream of consideration and is a touch lower.
raw["B1c_34"] = 5.80 + 2.50 * cons_latent + yf_lift["B1c_34"] * young_family \
                + 0.4 * RNG.normal(0, 1, N)

# Map every latent score to an integer 1-10.
def to_scale(x):
    return np.clip(np.rint(x), 1, 10).astype(int)

scores = {c: to_scale(raw[c]) for c in codes}
brand_df = pd.DataFrame({"respid": np.arange(1, N + 1)})
for c in codes:
    brand_df[c] = scores[c]
# Hold statement columns as object so text codes ("Don't know", scale anchors)
# can sit alongside the integers, the way a raw export does.
brand_df[codes] = brand_df[codes].astype(object)

# ---------------------------------------------------------------------------
# Inject realistic messiness
# ---------------------------------------------------------------------------
# 1) Correlated "Don't know". A latent "no view on the company" propensity makes
#    the corporate/citizenship items go unanswered together, so the missingness
#    is concentrated and overlapping rather than independent.
corp_ignorance = RNG.random(N)          # low value = weak view of the company
for c in predictors + ["B1c_33", "B1c_34"]:
    rate = dk_of[c]
    if c in CITIZENSHIP:                 # citizenship items: nested, correlated DK
        dk = corp_ignorance < rate
    else:                               # everyone else: independent, low DK
        dk = RNG.random(N) < rate
    brand_df.loc[dk, c] = "Don't know"

# 2) Straightliners: 93 respondents giving one value to all 34 statements.
straight_ids = RNG.choice(brand_df.index, size=93, replace=False)
flat_values = ([10] * 31 + [5] * 30 + [1] * 5
               + list(RNG.integers(2, 10, size=93 - 66)))
RNG.shuffle(flat_values)
for idx, val in zip(straight_ids, flat_values):
    brand_df.loc[idx, codes] = val

# 3) Store the scale endpoints as text, the way survey platforms export anchors.
def encode(v):
    if v == 1:  return "1 Completely disagree"
    if v == 10: return "10 Completely agree"
    return v

for c in codes:
    brand_df[c] = brand_df[c].map(encode)

# ---------------------------------------------------------------------------
# Assemble the demographics sheet and a codemap, then write the workbook
# ---------------------------------------------------------------------------
demo_df = pd.DataFrame({
    "respid": np.arange(1, N + 1),
    "D1": gender,
    "D2": age,
    "dD2": age_band,
    "D8": D8,
    **{c: kid_bands[c] for c in kid_band_cols},
    "region": region,
})

codemap_rows = [
    ("respid", "Respondent ID"),
    ("D1", "Gender"),
    ("D2", "Age"),
    ("dD2", "Age band"),
    ("D8", "Number of children under 16 (capped at 4+)"),
    ("D8b_1", "Child aged under 1"),
    ("D8b_2", "Child aged 1-3"),
    ("D8b_3", "Child aged 4-6"),
    ("D8b_4", "Child aged 7-9"),
    ("D8b_5", "Child aged 10-12"),
    ("D8b_6", "Child aged 13-15"),
    ("region", "Region"),
] + [(c, wording[c]) for c in codes]
codemap_df = pd.DataFrame(codemap_rows, columns=["Variable", "Question"])

out = "data/raw/freshfork_survey.xlsx"
with pd.ExcelWriter(out) as writer:
    demo_df.to_excel(writer, sheet_name="Demographics", index=False)
    brand_df.to_excel(writer, sheet_name="Brand Statements", index=False)
    codemap_df.to_excel(writer, sheet_name="Codemap", index=False)

print("Wrote", out)
print("Demographics:", demo_df.shape, "| Brand Statements:", brand_df.shape)
print("Young families:", int(young_family.sum()), "of", N)
