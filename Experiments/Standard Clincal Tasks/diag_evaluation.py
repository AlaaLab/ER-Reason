import pandas as pd

# ── Data sources ──────────────────────────────────────────────────────────────
# icd_10_codes.csv  — downloaded from PhysioNet alongside the ER-Reason dataset.
#                     Contains ground-truth ICD-10 codes per encounter.
#
# ccsr lookup file  — downloaded from AHRQ:
#                     https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp
#                     Use the "DXCCSR_v2025-1.CSV" or latest available release.
#
# diagnosis_results.csv — output of diagnosis.py (predicted_diagnosis column)

icd  = pd.read_csv('icd_10_codes.csv')
llm  = pd.read_csv('diagnosis_results.csv')
ccsr = pd.read_csv('DXCCSR_v2025-1.CSV')   # update filename to match your download


# ── Prep CCSR lookup (all categories per code) ────────────────────────────────
ccsr['icd_norm'] = (
    ccsr['ICD-10-CM Code']
    .astype(str).str.strip().str.upper()
    .str.replace(".", "", regex=False)
)
ccsr_lookup = (
    ccsr.groupby('icd_norm')['CCSR Category Description']
    .apply(list)
    .to_dict()
)


# ── Parse predicted_diagnosis into ICD code + name ────────────────────────────
# The diagnosis.py script stores the raw model response in predicted_diagnosis.
# This parser extracts the ICD-10 code from formats like:
#   "J18.9 Pneumonia, unspecified"  or  "[J18.9] Pneumonia, unspecified"

import re

def parse_prediction(text):
    if pd.isna(text) or text == 'Prediction failed':
        return None, None
    match = re.match(r'^\[?([A-Z][0-9]{2}\.?[0-9A-Z]*)\]?\s+(.*)', str(text).strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

llm[['predicted_icd_code', 'predicted_diagnosis_name']] = llm['predicted_diagnosis'].apply(
    lambda x: pd.Series(parse_prediction(x))
)


# ── Normalize ICD codes ───────────────────────────────────────────────────────
def normalize_icd(code):
    if pd.isna(code) or code == 'Prediction failed':
        return None
    return str(code).strip().upper().replace(".", "")


# ── Aggregate ground truth ICD codes per encounter ────────────────────────────
icd_grouped = (
    icd.groupby(['patientdurablekey', 'encounterkey'])
    .agg(
        gt_codes=(
            'value',
            lambda x: list(
                x.dropna().str.strip().str.upper()
                 .str.replace(".", "", regex=False).unique()
            )
        ),
        gt_displaystrings=('displaystring', lambda x: list(x.dropna().unique()))
    )
    .reset_index()
)


# ── Merge predictions with ground truth ───────────────────────────────────────
merged = llm.merge(icd_grouped, on=['patientdurablekey', 'encounterkey'], how='left')
merged['pred_normalized'] = merged['predicted_icd_code'].apply(normalize_icd)


# ── ICD exact match ───────────────────────────────────────────────────────────
def is_correct(row):
    if row['pred_normalized'] is None or not isinstance(row['gt_codes'], list):
        return False
    return row['pred_normalized'] in row['gt_codes']

def is_correct_prefix(row):
    if row['pred_normalized'] is None or not isinstance(row['gt_codes'], list):
        return False
    pred_prefix = row['pred_normalized'][:3]
    return any(gt[:3] == pred_prefix for gt in row['gt_codes'])

merged['correct']        = merged.apply(is_correct, axis=1)
merged['correct_prefix'] = merged.apply(is_correct_prefix, axis=1)


# ── Map CCSR categories ───────────────────────────────────────────────────────
merged['pred_ccsr'] = merged['pred_normalized'].map(ccsr_lookup)

merged['gt_ccsr'] = merged['gt_codes'].apply(
    lambda codes: list({
        cat
        for c in codes
        for cat in (ccsr_lookup.get(c) or [])
    }) if isinstance(codes, list) else []
)


# ── CCSR accuracy ─────────────────────────────────────────────────────────────
def ccsr_correct(row):
    pred = row['pred_ccsr']
    gt   = row['gt_ccsr']
    if not isinstance(gt, list) or len(gt) == 0:
        return False
    if not isinstance(pred, list) or len(pred) == 0:
        return False
    return bool(set(pred) & set(gt))

merged['ccsr_correct'] = merged.apply(ccsr_correct, axis=1)


# ── Results ───────────────────────────────────────────────────────────────────
def print_results(df, label=""):
    total         = len(df)
    has_gt_ccsr   = df['gt_ccsr'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
    has_pred_ccsr = df['pred_ccsr'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
    evaluable     = df[df['gt_ccsr'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    header = f" {label} " if label else ""
    print(f"\n{'=' * 55}")
    print(f"{'ICD-10 Exact Match':^55}" + (f" [{label}]" if label else ""))
    print(f"{'=' * 55}")
    print(f"Accuracy:                  {df['correct'].mean():.2%}  ({df['correct'].sum()} / {total})")
    print(f"Category Accuracy (3-char):{df['correct_prefix'].mean():.2%}  ({df['correct_prefix'].sum()} / {total})")

    print(f"\n{'=' * 55}")
    print(f"{'CCSR Clinical Category':^55}" + (f" [{label}]" if label else ""))
    print(f"{'=' * 55}")
    print(f"Accuracy (all rows):       {df['ccsr_correct'].mean():.2%}  ({df['ccsr_correct'].sum()} / {total})")
    print(f"GT rows with CCSR mapping: {has_gt_ccsr} / {total} ({has_gt_ccsr/total:.1%})")
    print(f"Pred rows with CCSR:       {has_pred_ccsr} / {total} ({has_pred_ccsr/total:.1%})")
    print(f"Accuracy (mappable GT):    {evaluable['ccsr_correct'].mean():.2%}  ({evaluable['ccsr_correct'].sum()} / {len(evaluable)})")


# Print results per condition (zero_shot, step_back) and overall
for condition, grp in merged.groupby('condition'):
    print_results(grp, label=condition)

print_results(merged, label="all")


# ── Sample misses ─────────────────────────────────────────────────────────────
misses = merged[~merged['correct']][[
    'patientdurablekey', 'encounterkey', 'condition',
    'gt_codes', 'pred_normalized',
    'gt_displaystrings', 'predicted_diagnosis_name',
    'gt_ccsr', 'pred_ccsr', 'ccsr_correct'
]]
pd.set_option('display.max_colwidth', 60)
print(f"\nSample misses (first 10):")
print(misses.head(10).to_string(index=False))