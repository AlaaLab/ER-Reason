import os
import re
import time
import random
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MODEL_NAME = "deepseek/deepseek-r1"

MAX_RETRIES     = 5
N_WORKERS       = 10
PRINCIPLES_FILE = "disposition_principles.txt"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an experienced Emergency Department physician."

VALID_DISPOSITIONS = [
    'Discharge', 'Admit', 'Observation', 'Transfer to Another Facility',
    'AMA', 'OR Admit', 'LWBS after Triage', 'Send to L&D',
    'Expired', 'Eloped'
]


# ── Note cleaning ─────────────────────────────────────────────────────────────
# Truncates ED note before "Final Disposition and ED Course" and redacts the
# disposition label to prevent label leakage.

def clean_ed_note(row):
    note        = row.get('ED_Provider_Notes_Text', '')
    disposition = row.get('eddisposition', '')

    if pd.isna(note) or note == '':
        return None

    note = str(note)
    match = re.search(r'(?i)final\s+disposition\s+and\s+ed\s+course', note)
    if match:
        note = note[:match.start()].strip()

    if not pd.isna(disposition) and disposition != '':
        note = re.sub(re.escape(str(disposition)), '[REDACTED]', note, flags=re.IGNORECASE)

    return note if note.strip() else None


# ── Step-back: Call 1 — fetch principles once ─────────────────────────────────
def get_stepback_principles(principles_file=PRINCIPLES_FILE):
    """Fetch ED disposition principles once and cache to disk for reuse."""
    if os.path.exists(principles_file):
        with open(principles_file) as f:
            principles = f.read().strip()
        print(f"Loaded cached principles from {principles_file}")
        return principles

    print("Fetching ED disposition principles (step-back Call 1)...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "What are the key clinical principles and criteria that determine ED disposition? "
                "Please describe what patient factors, diagnoses, vital sign patterns, social "
                "circumstances, and clinical findings typically lead to each of the following "
                "outcomes: Discharge, Admit, Observation, Transfer to Another Facility, AMA, "
                "OR Admit, LWBS after Triage, Send to L&D, Eloped."
            )}
        ],
        temperature=0.1,
        max_tokens=800,
        extra_body={"provider": {"zdr": True}},
    )
    principles = response.choices[0].message.content.strip()

    with open(principles_file, "w") as f:
        f.write(principles)
    print(f"Principles saved to {principles_file}")
    return principles


# ── Prompt builders ───────────────────────────────────────────────────────────
def build_patient_fields(row):
    """Shared patient field block used by both zero-shot and step-back prompts."""
    fields = ""
    if 'Age' in row and not pd.isna(row['Age']):
        fields += f"Age: {row['Age']}\n"
    if 'sex' in row and not pd.isna(row['sex']):
        fields += f"Sex: {row['sex']}\n"
    fields += f"Chief Complaint: {row['primarychiefcomplaintname']}\n"
    if 'primaryeddiagnosisname' in row and not pd.isna(row['primaryeddiagnosisname']):
        fields += f"Primary ED Diagnosis: {row['primaryeddiagnosisname']}\n"
    cleaned_note = clean_ed_note(row)
    if cleaned_note:
        fields += f"\nED Provider Note:\n{cleaned_note}\n"
    return fields


def build_zero_shot_prompt(row):
    """Zero-shot: patient fields + label selection."""
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt  = "Predict the most likely ED disposition for this patient.\n\n"
    prompt += build_patient_fields(row)
    prompt += "\nChoose ONLY from the following options:\n"
    prompt += ", ".join(f"'{d}'" for d in VALID_DISPOSITIONS)
    prompt += "\n\nRespond with ONLY ONE of these options on the last line. No explanation."
    return prompt


def build_stepback_prompt(row, principles):
    """Step-back Call 2: inject retrieved principles then patient fields."""
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt  = "Using the following clinical principles:\n"
    prompt += f"{principles}\n\n"
    prompt += "Now predict the most likely ED disposition for this patient:\n\n"
    prompt += build_patient_fields(row)
    prompt += "\nChoose ONLY from the following options:\n"
    prompt += ", ".join(f"'{d}'" for d in VALID_DISPOSITIONS)
    prompt += "\n\nRespond with ONLY ONE of these options on the last line. No explanation."
    return prompt


# ── API call with retry ───────────────────────────────────────────────────────
def call_model(prompt, encounterkey, max_retries=MAX_RETRIES):
    backoff = 2
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,  # remove this line for o4-mini
                max_tokens=100,   # enough for a disposition label; reasoning models
                                  # consume tokens internally — increase if empty responses
                extra_body={"provider": {"zdr": True}},  # Zero Data Retention
            )

            raw = response.choices[0].message.content
            if raw is None:
                return "Prediction failed"

            raw = raw.strip()
            # Check last line first (model may add brief reasoning before label)
            last_line = raw.split('\n')[-1].strip()
            for disposition in VALID_DISPOSITIONS:
                if disposition.lower() in last_line.lower():
                    return disposition
            # Fallback: search full response
            for disposition in VALID_DISPOSITIONS:
                if disposition.lower() in raw.lower():
                    return disposition

            print(f"Unmatched response for {encounterkey}: '{raw[:100]}'")
            return raw

        except Exception as e:
            err = str(e)
            if '429' in err or 'rate_limit' in err.lower() or 'Connection' in err:
                wait = backoff + random.uniform(0, 1)
                print(f"Rate limit / connection error. Waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
            else:
                print(f"Error on {encounterkey}: {e}")
                return "Prediction failed"

    return "Prediction failed"


# ── Checkpointed parallel runner ──────────────────────────────────────────────
def run_condition(df, condition, prompt_fn, checkpoint_file):
    """
    Run one condition (zero_shot or step_back) with checkpointing.
    prompt_fn: callable that takes a row and returns a prompt string.
    """
    if os.path.exists(checkpoint_file):
        processed_df  = pd.read_csv(checkpoint_file)
        processed_ids = set(processed_df['encounterkey'].astype(str).tolist())
        print(f"[{condition}] Resuming from checkpoint: {len(processed_ids)} already done")
    else:
        processed_df  = pd.DataFrame()
        processed_ids = set()

    remaining     = df[~df['encounterkey'].astype(str).isin(processed_ids)]
    total         = len(remaining)
    batch_size    = 50
    total_batches = (total + batch_size - 1) // batch_size
    start_time    = time.time()
    processed_count = 0

    print(f"[{condition}] {total} records remaining | {N_WORKERS} workers")

    for batch_num in range(total_batches):
        batch = remaining.iloc[batch_num * batch_size:(batch_num + 1) * batch_size]
        print(f"\n[{condition}] Batch {batch_num+1}/{total_batches} ({len(batch)} records)")

        batch_results = {}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(
                    call_model,
                    prompt_fn(row),
                    str(row['encounterkey'])
                ): str(row['encounterkey'])
                for _, row in batch.iterrows()
                if prompt_fn(row) is not None
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                ek = futures[future]
                batch_results[ek] = future.result()

        rows = []
        for _, row in batch.iterrows():
            ek       = str(row['encounterkey'])
            row_copy = row.to_dict()
            row_copy['condition']             = condition
            row_copy['predicted_disposition'] = batch_results.get(ek, "Prediction failed")
            rows.append(row_copy)

        processed_df    = pd.concat([processed_df, pd.DataFrame(rows)], ignore_index=True)
        processed_ids.update(str(r['encounterkey']) for r in rows)
        processed_count += len(rows)
        processed_df.to_csv(checkpoint_file, index=False)

        elapsed = time.time() - start_time
        rps     = processed_count / elapsed
        eta     = (total - processed_count) / rps if rps > 0 else float('inf')
        print(f"[{condition}] Progress: {processed_count}/{total} | {rps:.3f} rec/s | ETA: {eta/60:.1f} min")

    return processed_df


# ── Accuracy ──────────────────────────────────────────────────────────────────
def calculate_accuracy(df, true_col='eddisposition'):
    for condition, grp in df.groupby('condition'):
        valid   = grp[grp['predicted_disposition'].isin(VALID_DISPOSITIONS)].copy()
        invalid = len(grp) - len(valid)
        if invalid > 0:
            print(f"[{condition}] Warning: {invalid} rows excluded")
        matches  = (valid['predicted_disposition'] == valid[true_col]).sum()
        total    = len(valid)
        accuracy = matches / total if total > 0 else 0
        print(f"\n[{condition}] Overall Accuracy: {matches}/{total} = {accuracy:.2%}")
        for disp in VALID_DISPOSITIONS:
            lvl = valid[valid[true_col] == disp]
            if len(lvl) > 0:
                acc = (lvl['predicted_disposition'] == lvl[true_col]).sum() / len(lvl)
                print(f"  {disp:<30}: {acc:.2%} ({len(lvl)} records)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load your dataset — must contain: encounterkey, primarychiefcomplaintname,
    # ED_Provider_Notes_Text, and optionally: Age, sex, primaryeddiagnosisname,
    # eddisposition (for evaluation)
    # df = pd.read_csv("your_data.csv")

    print(f"Loaded {len(df)} records\n")

    # ── Zero-shot ─────────────────────────────────────────────────────────────
    zs_df = run_condition(
        df,
        condition="zero_shot",
        prompt_fn=build_zero_shot_prompt,
        checkpoint_file="disposition_zero_shot_checkpoint.csv",
    )

    # ── Step-back ─────────────────────────────────────────────────────────────
    principles = get_stepback_principles()
    sb_df = run_condition(
        df,
        condition="step_back",
        prompt_fn=lambda row: build_stepback_prompt(row, principles),
        checkpoint_file="disposition_step_back_checkpoint.csv",
    )

    # ── Combine and save ──────────────────────────────────────────────────────
    results_df = pd.concat([zs_df, sb_df], ignore_index=True)
    results_df.to_csv("disposition_results.csv", index=False)
    print("\nSaved to disposition_results.csv")

    calculate_accuracy(results_df, true_col='eddisposition')