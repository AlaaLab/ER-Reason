import pandas as pd
from statsmodels.stats.proportion import proportion_confint

# ── Data loading ──────────────────────────────────────────────────────────────
# acuity_results.csv      — output of acuity.py
#                           must contain: patientdurablekey, encounterkey,
#                           acuitylevel, predicted_acuity, condition
#
# disposition_results.csv — output of disposition.py
#                           must contain: patientdurablekey, encounterkey,
#                           eddisposition, predicted_disposition, condition
#
# diagnosis_results.csv   — output of diagnosis.py
#                           must contain: patientdurablekey, encounterkey,
#                           predicted_diagnosis, condition
#
# icd_10_codes.csv        — downloaded from PhysioNet alongside ER-Reason dataset
#                           contains ground-truth ICD-10 codes per encounter
#
# ccsr reference file     — downloaded from AHRQ:
#                           https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp
#                           use "DXCCSR_v2025-1.CSV" or latest available release

# llm_acuity      = pd.read_csv('acuity_results.csv')
# llm_disposition = pd.read_csv('disposition_results.csv')
# llm_diagnosis   = pd.read_csv('diagnosis_results.csv')
# icd             = pd.read_csv('icd_10_codes.csv')
# ccsr            = pd.read_csv('DXCCSR_v2025-1.CSV')   # update filename to match your download

KEYS = ['patientdurablekey', 'encounterkey']

# ── Internal mappings ─────────────────────────────────────────────────────────
ESI_MAP = {
    'Immediate':   'high',
    'Emergent':    'high',
    'Urgent':      'mid',
    'Less Urgent': 'low',
    'Non-Urgent':  'low',
}

DISP_COLLAPSE = {
    'Admit':                        'admit',
    'OR Admit':                     'admit',
    'Observation':                  'admit',
    'Discharge':                    'discharge',
    'Transfer to Another Facility': 'transfer',
    'AMA':                          'other',
    'Eloped':                       'other',
    'Expired':                      'other',
    'LWBS after Triage':            'other',
    'Send to L&D':                  'other',
    'Dismissed - Never Arrived':    'other',
}


# ── ICD-10 normalization ──────────────────────────────────────────────────────
def normalize_icd(code):
    if pd.isna(code) or str(code).strip() in ('', 'Prediction failed'):
        return None
    return str(code).strip().upper().replace('.', '')


# ── Parse predicted_diagnosis → ICD code ─────────────────────────────────────
# diagnosis.py stores the raw model response; extract the ICD-10 code here.
import re

def parse_icd_from_prediction(text):
    if pd.isna(text) or text == 'Prediction failed':
        return None
    match = re.match(r'^\[?([A-Z][0-9]{2}\.?[0-9A-Z]*)\]?', str(text).strip())
    return match.group(1).strip() if match else None


# ── CCSR lookup → inpatient / outpatient flag per ICD ────────────────────────
ccsr['icd_norm'] = (
    ccsr['ICD-10-CM Code']
    .astype(str).str.strip().str.upper()
    .str.replace('.', '', regex=False)
)
ccsr_inpatient  = ccsr.set_index('icd_norm')['Inpatient Default CCSR (Y/N/X)'].to_dict()
ccsr_outpatient = ccsr.set_index('icd_norm')['Outpatient Default CCSR (Y/N/X)'].to_dict()


# ── Ground truth ICD codes per encounter ─────────────────────────────────────
icd_grouped = (
    icd.groupby(KEYS)
    .agg(
        gt_codes=(
            'value',
            lambda x: list(
                x.dropna().str.strip().str.upper()
                 .str.replace('.', '', regex=False).unique()
            )
        )
    )
    .reset_index()
)


# ── Ground truth bucket assignment ───────────────────────────────────────────
def assign_bucket(row):
    esi  = ESI_MAP.get(row['acuitylevel'])
    disp = DISP_COLLAPSE.get(row['eddisposition'])
    if esi == 'high':
        return 'high'
    elif esi == 'low':
        return 'low'
    elif esi == 'mid':
        if disp == 'admit':
            return 'high'
        elif disp == 'discharge':
            return 'low'
    return None  # ESI 3 transfer/AMA/other and unspecified → excluded


# ── Scoring function ──────────────────────────────────────────────────────────
def score(row):
    if row['bucket'] == 'high':
        return int(
            row['pred_disp'] == 'admit' and
            row['pred_inpatient'] == 'Y'
        )
    elif row['bucket'] == 'low':
        return int(
            row['pred_disp'] == 'discharge' and
            row['pred_outpatient'] == 'Y'
        )
    return 0


# ── Reporting with Wilson CIs ─────────────────────────────────────────────────
def report_ci(label, subset):
    n       = len(subset)
    correct = subset['correct'].sum()
    rate    = subset['correct'].mean()
    ci      = proportion_confint(correct, n, alpha=0.05, method='wilson')
    print(f"  {label}: {rate:.2%} ({ci[0]:.2%}–{ci[1]:.2%})  ({correct}/{n})")


# ── Run evaluation per condition ──────────────────────────────────────────────
def run_cross_stage(llm_acuity, llm_disposition, llm_diagnosis, condition):
    acuity = llm_acuity[llm_acuity['condition'] == condition]
    dispos = llm_disposition[llm_disposition['condition'] == condition]
    diagn  = llm_diagnosis[llm_diagnosis['condition'] == condition]

    # Extract predicted ICD code from raw model response
    diagn = diagn.copy()
    diagn['predicted_icd_code'] = diagn['predicted_diagnosis'].apply(parse_icd_from_prediction)

    # Build ground truth bucket table
    gt = acuity[KEYS + ['acuitylevel', 'predicted_acuity']].copy()
    gt = gt.merge(
        dispos[KEYS + ['eddisposition', 'predicted_disposition']],
        on=KEYS, how='inner'
    )
    gt['bucket'] = gt.apply(assign_bucket, axis=1)
    gt = gt[gt['bucket'].notna()]

    # Merge predicted diagnosis + ground truth ICD
    merged = gt.merge(
        diagn[KEYS + ['predicted_icd_code']], on=KEYS, how='left'
    ).merge(
        icd_grouped, on=KEYS, how='left'
    )

    # Normalize and map predicted ICD to CCSR flags
    merged['pred_icd_norm']   = merged['predicted_icd_code'].apply(normalize_icd)
    merged['pred_inpatient']  = merged['pred_icd_norm'].map(ccsr_inpatient)
    merged['pred_outpatient'] = merged['pred_icd_norm'].map(ccsr_outpatient)
    merged['pred_disp']       = merged['predicted_disposition'].map(DISP_COLLAPSE)
    merged['correct']         = merged.apply(score, axis=1)

    high = merged[merged['bucket'] == 'high']
    low  = merged[merged['bucket'] == 'low']
    excluded = len(gt) - len(high) - len(low)

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition.upper()}")
    print(f"{'='*60}")
    print(f"  Total encounters: {len(merged)} | Excluded (ESI3 other/transfer): {excluded}")
    report_ci("ESI 1-2 + ESI3(Admitted)   — Admit ∩ CCSR Inpatient   ", high)
    report_ci("ESI 4-5 + ESI3(Discharged) — Discharge ∩ CCSR Outpatient", low)

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    conditions = llm_acuity['condition'].unique()

    all_results = {}
    for condition in conditions:
        merged = run_cross_stage(llm_acuity, llm_disposition, llm_diagnosis, condition)
        all_results[condition] = merged