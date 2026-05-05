import os
import time
import random
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
# Set your OpenRouter API key as an environment variable:
#   export OPENROUTER_API_KEY="sk-or-..."
# or paste it directly here (not recommended for shared/public code).

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Swap in any OpenRouter-hosted model below. Examples:
#   "openai/gpt-5.2-20251211"
#   "openai/o4-mini"                     # remove temperature parameter for this model
#   "deepseek/deepseek-r1"
#   "google/gemini-2.5-flash"
#   "anthropic/claude-sonnet-4-5"        # add extra_body={"thinking": {"type": "enabled",
#                                        #   "budget_tokens": 10000}} to enable thinking mode
#   "microsoft/phi-4"
MODEL_NAME  = "deepseek/deepseek-r1"

CONDITIONS  = ['baseline', 'single_oracle', 'full_oracle']
MAX_RETRIES = 5

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an experienced emergency medicine physician reasoning through a clinical case.
You will be given a patient presentation and a fixed list of diagnoses to consider.
At each step, you will receive a new piece of evidence paired with one of those diagnoses.
Your job is to update your belief about that diagnosis based on the evidence, then re-rank the FULL list.

You MUST only use the diagnoses from the fixed list provided. Do not add, rename, or invent any diagnoses.

Respond ONLY in the following JSON format. Do not include any explanation outside the JSON.
{
  "dx_updates": [
    {"diagnosis": "<exact dx name from list>", "update": <-2|-1|0|1|2>, "rationale": "<one sentence>"}
  ],
  "ranked_differential": ["<most likely>", "<second>", ..., "<least likely>"],
  "reasoning_summary": "<one sentence overall reasoning>"
}

Update score semantics:
+2 = this evidence strongly increases the likelihood of this diagnosis
+1 = this evidence mildly increases the likelihood of this diagnosis
 0 = this evidence does not meaningfully change the likelihood
-1 = this evidence mildly decreases the likelihood of this diagnosis
-2 = this evidence strongly decreases the likelihood of this diagnosis

FINAL STEP ONLY — also include:
  "final_diagnosis": "<single most likely diagnosis from the fixed list>"
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_dx(dx):
    return str(dx).strip().rstrip(':').strip()


def normalize(s):
    s = clean_dx(s).lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def snap_to_pool(dx_str, pool_normalized, pool_original):
    n = normalize(dx_str)
    if n in pool_normalized:
        return pool_original[pool_normalized.index(n)]
    for i, pn in enumerate(pool_normalized):
        if n in pn or pn in n:
            return pool_original[i]
    return None


def fix_ranked_differential(ranked, all_dxs):
    pool_norm = [normalize(dx) for dx in all_dxs]
    pool_orig = list(all_dxs)
    corrected = []
    for dx in ranked:
        snapped = snap_to_pool(dx, pool_norm, pool_orig)
        if snapped is None:
            return None
        corrected.append(snapped)
    if sorted([normalize(x) for x in corrected]) != sorted(pool_norm):
        return None
    return corrected


def get_differentials(row):
    dxs = []
    for t in range(1, 6):
        dx = row.get(f'differential_{t}')
        if not pd.isna(dx) and str(dx).strip() != '':
            dxs.append(clean_dx(dx))
    return dxs


def get_evidence_steps(row):
    steps = []
    for t in range(1, 6):
        dx = row.get(f'differential_{t}')
        ev = row.get(f'evidence_{t}')
        if not pd.isna(dx) and not pd.isna(ev) and \
           str(dx).strip() != '' and str(ev).strip() != '':
            steps.append((t, clean_dx(dx), str(ev).strip()))
    return steps


def get_rationale(gt, encounterkey, differential):
    match = gt[
        (gt['encounterkey'] == encounterkey) &
        (gt['differential'].str.lower().str.strip() == differential.lower().strip())
    ]
    if len(match) > 0 and not pd.isna(match.iloc[0]['rationale']):
        return str(match.iloc[0]['rationale']).strip()
    return None


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(one_sentence, all_dxs, current_dx, evidence,
                 prior_ranking=None, timestep=1, is_final=False,
                 current_rationale=None, prior_rationales=None):
    prompt  = f"=== PATIENT CASE ===\n{one_sentence}\n"
    prompt += f"\n=== FIXED DIAGNOSIS LIST (re-rank these at every step) ===\n"
    for dx in all_dxs:
        prompt += f"- {dx}\n"
    prompt += f"\n=== NEW EVIDENCE (Step {timestep}) ===\n{evidence}\n"
    prompt += f"\n=== DIAGNOSIS THIS EVIDENCE RELATES TO ===\n{current_dx}\n"

    if prior_rationales:
        prompt += f"\n=== PHYSICIAN RATIONALE FOR PRIOR STEPS ===\n"
        for step_t, dx, rationale in prior_rationales:
            prompt += f"Step {step_t} ({dx}): {rationale}\n"

    if current_rationale:
        prompt += f"\n=== PHYSICIAN RATIONALE FOR THIS STEP ===\n{current_rationale}\n"

    if prior_ranking:
        prompt += f"\n=== YOUR PREVIOUS RANKING (Step {timestep - 1}) ===\n"
        for i, dx in enumerate(prior_ranking, 1):
            prompt += f"{i}. {dx}\n"
        prompt += "\nGiven the evidence above, update your belief and re-rank the full list."
    else:
        prompt += "\nThis is the first piece of evidence. Provide your initial belief score and rank the full list."

    if is_final:
        prompt += "\n\nThis is the FINAL step. Also provide 'final_diagnosis' — your single best diagnosis from the fixed list."

    return prompt


# ── Response parsing and validation ──────────────────────────────────────────
def parse_response(raw_text):
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def is_valid_response(parsed, all_dxs, is_final=False):
    if parsed is None:
        return False
    if 'dx_updates' not in parsed or 'ranked_differential' not in parsed:
        return False
    if not isinstance(parsed['dx_updates'], list) or len(parsed['dx_updates']) == 0:
        return False
    if not isinstance(parsed['ranked_differential'], list) or len(parsed['ranked_differential']) == 0:
        return False
    for item in parsed['dx_updates']:
        if 'update' not in item or item['update'] not in [-2, -1, 0, 1, 2]:
            return False
    corrected = fix_ranked_differential(parsed['ranked_differential'], all_dxs)
    if corrected is None:
        pool_norm   = {normalize(dx) for dx in all_dxs}
        ranked_norm = {normalize(dx) for dx in parsed['ranked_differential']}
        extra   = ranked_norm - pool_norm
        missing = pool_norm - ranked_norm
        if extra:   print(f"    Hallucinated dx: {extra}")
        if missing: print(f"    Missing dx: {missing}")
        return False
    parsed['ranked_differential'] = corrected
    if is_final:
        fd = parsed.get('final_diagnosis', '')
        if not fd or not str(fd).strip():
            parsed['final_diagnosis'] = corrected[0]
        else:
            pool_norm = [normalize(dx) for dx in all_dxs]
            pool_orig = list(all_dxs)
            snapped   = snap_to_pool(str(fd), pool_norm, pool_orig)
            if snapped is None:
                return False
            parsed['final_diagnosis'] = snapped
    return True


# ── API call with retry ───────────────────────────────────────────────────────
def call_model(prompt, encounterkey, timestep, all_dxs, condition, is_final=False):
    backoff = 2
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,  # remove this line for o4-mini
                max_tokens=4000,  # SCT requires structured JSON — reasoning models need
                                  # extra budget for chain-of-thought before the response
                extra_body={"provider": {"zdr": True}},  # Zero Data Retention
            )
            raw_text = response.choices[0].message.content.strip()
            parsed   = parse_response(raw_text)
            if is_valid_response(parsed, all_dxs, is_final=is_final):
                return raw_text, parsed, None
            print(f"  [{condition} | {encounterkey} step {timestep}] Invalid response "
                  f"(attempt {attempt + 1}), retrying...")
            time.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 30)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = backoff + random.uniform(0, 2)
                print(f"  [{condition} | {encounterkey} step {timestep}] Rate limit, "
                      f"waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
            elif any(code in err_str for code in ["500", "502", "503"]):
                wait = backoff + random.uniform(0, 1)
                print(f"  [{condition} | {encounterkey} step {timestep}] Server error, "
                      f"waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                backoff = min(backoff * 2, 30)
            else:
                print(f"  [{condition} | {encounterkey} step {timestep}] Error: {e}")
                return None, None, str(e)
    return None, None, "max_retries_exceeded"


# ── Per-encounter runner ──────────────────────────────────────────────────────
def process_encounter(row, gt, condition):
    encounterkey     = row['encounterkey']
    one_sentence     = row['One_Sentence_Extracted']
    all_dxs          = get_differentials(row)
    steps            = get_evidence_steps(row)
    results          = []
    prior_ranking    = None
    prior_rationales = []

    for idx, (t, current_dx, evidence) in enumerate(steps):
        is_final = (idx == len(steps) - 1)

        rationale = get_rationale(gt, encounterkey, current_dx)

        current_rationale = rationale if condition in ('single_oracle', 'full_oracle') else None
        prev_rationales   = prior_rationales if condition == 'full_oracle' else None

        prompt = build_prompt(
            one_sentence, all_dxs, current_dx, evidence,
            prior_ranking=prior_ranking,
            timestep=t,
            is_final=is_final,
            current_rationale=current_rationale,
            prior_rationales=prev_rationales if prev_rationales else None,
        )

        raw_text, parsed, error = call_model(
            prompt, encounterkey, t, all_dxs, condition, is_final=is_final
        )

        ranked = parsed.get('ranked_differential') if parsed else None

        results.append({
            'condition':           condition,
            'encounterkey':        encounterkey,
            'timestep':            t,
            'is_final_step':       is_final,
            'differential':        current_dx,
            'evidence':            evidence,
            'physician_rationale': rationale,
            'all_dxs':             json.dumps(all_dxs),
            'raw_response':        raw_text,
            'dx_updates_json':     json.dumps(parsed.get('dx_updates')) if parsed else None,
            'ranked_differential': json.dumps(ranked) if ranked else None,
            'reasoning_summary':   parsed.get('reasoning_summary') if parsed else None,
            'final_diagnosis':     parsed.get('final_diagnosis') if (parsed and is_final) else None,
            'parse_error':         error,
        })

        if ranked:
            prior_ranking = ranked
        if rationale:
            prior_rationales.append((t, current_dx, rationale))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data — update paths as needed
    # annotated_sct.csv    — ER-Reason dataset with One_Sentence_Extracted and
    #                        differential_1..5 / evidence_1..5 columns
    # sct-annotations.csv  — SCT case list (used to filter encounterkeys)
    # sct_cleaned_annotations.csv — physician rationales (gt) with columns:
    #                        encounterkey, differential, rationale
    test = pd.read_csv("annotated_sct.csv")
    sct  = pd.read_csv("sct-annotations.csv")
    gt   = pd.read_csv("sct_cleaned_annotations.csv")

    sct_keys = sct['encounterkey'].unique()
    test     = test[test['encounterkey'].isin(sct_keys)].copy()

    keep_cols = [
        'encounterkey', 'One_Sentence_Extracted',
        'differential_1', 'evidence_1',
        'differential_2', 'evidence_2',
        'differential_3', 'evidence_3',
        'differential_4', 'evidence_4',
        'differential_5', 'evidence_5',
    ]
    test = test[keep_cols].drop_duplicates('encounterkey').reset_index(drop=True)
    print(f"Running SCT on {len(test)} encounters")
    print(f"Conditions: {CONDITIONS}\n")

    for condition in CONDITIONS:
        print(f"\n{'='*50}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*50}")

        checkpoint_file = f"sct_{condition}_checkpoint.csv"

        if os.path.exists(checkpoint_file):
            done_df  = pd.read_csv(checkpoint_file)
            done_eks = set(done_df['encounterkey'].unique())
            print(f"  Resuming — {len(done_eks)} encounters already done")
        else:
            done_df  = pd.DataFrame()
            done_eks = set()

        remaining = test[~test['encounterkey'].isin(done_eks)]
        print(f"  Remaining: {len(remaining)} encounters\n")

        start_time = time.time()

        for i, (_, row) in enumerate(tqdm(remaining.iterrows(),
                                          total=len(remaining),
                                          desc=condition)):
            new_results = process_encounter(row, gt, condition)
            step_df     = pd.DataFrame(new_results)

            if os.path.exists(checkpoint_file):
                step_df.to_csv(checkpoint_file, mode='a', header=False, index=False)
            else:
                step_df.to_csv(checkpoint_file, index=False)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate    = (i + 1) / elapsed
                eta_min = (len(remaining) - i - 1) / rate / 60
                print(f"  {i + 1}/{len(remaining)} | ETA {eta_min:.1f} min")

    # ── Combine and save ──────────────────────────────────────────────────────
    final_df = pd.concat(
        [pd.read_csv(f"sct_{c}_checkpoint.csv") for c in CONDITIONS],
        ignore_index=True
    )

    output_file = "sct_results.csv"
    final_df.to_csv(output_file, index=False)

    print(f"\nDone. Results saved to {output_file}")
    print(f"Total rows:        {len(final_df)}")
    print(f"Unique encounters: {final_df['encounterkey'].nunique()}")
    print(f"\nRows per condition:")
    print(final_df.groupby('condition').size())
    print(f"\nErrors per condition:")
    print(final_df.groupby('condition')['parse_error'].apply(lambda x: x.notna().sum()))