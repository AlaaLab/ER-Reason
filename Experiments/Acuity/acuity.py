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
MODEL_NAME = "microsoft/phi-4"

MAX_RETRIES = 5
N_WORKERS   = 10

PRINCIPLES_FILE = "acuity_principles.txt"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an experienced Emergency Department triage nurse."

VALID_ACUITY = ['Immediate', 'Emergent', 'Urgent', 'Less Urgent', 'Non-Urgent']


# ── Vital signs extraction ────────────────────────────────────────────────────
# Vital_Signs is not a standard column in the ER-Reason dataset.
# Run extract_vital_signs() on ED_Provider_Notes_Text before running the experiment.

def extract_vital_signs(text):
    if not isinstance(text, str):
        return "No vital signs available"

    # Pattern 1: Standard "Triage Vital Signs:" block up to next section header
    match = re.search(r"Triage Vital Signs:.*?(?=HENT:)", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Pattern 2: Any vital signs header up to a physical exam section
    match = re.search(
        r"(?:Triage Vital Signs|Vital Signs)[\s:]*.*?"
        r"(?=(?:HENT|Head|Eyes|Cardiovascular|Pulmonary|Constitutional|Physical Exam):)",
        text, re.DOTALL
    )
    if match:
        return match.group(0).strip()

    # Pattern 3: Named vital sign fields
    vitals = re.findall(r"(?:BP|Heart Rate|Pulse|Temp|Resp|SpO2|Temperature)[\s:].{1,150}", text)
    if vitals:
        return "Vital Signs: " + " ".join(vitals)

    # Pattern 4: Abbreviated vital sign labels with numeric values
    vitals = re.findall(r"(?:BP|HR|RR|T|Temp|O2)[\s:]*\d+(?:[\/\.\-]\d+)?(?:\s*[%℃℉]?)+", text)
    if vitals:
        return "Vital Signs: " + " ".join(vitals)

    # Pattern 5: Blood pressure pattern as last resort
    match = re.search(r"\d{2,3}\/\d{2,3}", text)
    if match:
        start = max(0, match.start() - 100)
        end   = min(len(text), match.end() + 100)
        return "Possible Vital Signs: " + text[start:end]

    return "No vital signs available"


# ── Step-back: Call 1 — fetch principles once ─────────────────────────────────
def get_stepback_principles(principles_file=PRINCIPLES_FILE):
    """Fetch ESI triage principles once and cache to disk for reuse."""
    if os.path.exists(principles_file):
        with open(principles_file) as f:
            principles = f.read().strip()
        print(f"Loaded cached principles from {principles_file}")
        return principles

    print("Fetching ESI triage principles (step-back Call 1)...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "What are the key clinical principles and criteria that differentiate each ESI "
                "triage level? Please describe the vital sign thresholds, symptom patterns, "
                "chief complaint characteristics, and risk factors that indicate each of the "
                "following acuity levels: Immediate, Emergent, Urgent, Less Urgent, Non-Urgent."
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
    if 'firstrace' in row and not pd.isna(row['firstrace']):
        fields += f"Race: {row['firstrace']}\n"
    fields += f"Chief Complaint: {row['primarychiefcomplaintname']}\n"
    if 'Vital_Signs' in row and not pd.isna(row['Vital_Signs']):
        fields += f"Vital Signs: {row['Vital_Signs']}\n"
    return fields


def build_zero_shot_prompt(row):
    """Zero-shot: patient fields + label selection."""
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt  = "Predict the emergency department acuity level for this patient.\n\n"
    prompt += build_patient_fields(row)
    prompt += "\nSelect the most appropriate acuity level from the following options ONLY:\n"
    prompt += ", ".join(f"'{a}'" for a in VALID_ACUITY)
    prompt += "\n\nRespond with ONLY ONE of these five options. No explanation."
    return prompt


def build_stepback_prompt(row, principles):
    """Step-back Call 2: inject retrieved principles then patient fields."""
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt  = "Using the following clinical triage principles:\n"
    prompt += f"{principles}\n\n"
    prompt += "Now assign the acuity level for this patient:\n\n"
    prompt += build_patient_fields(row)
    prompt += "\nSelect the most appropriate acuity level from the following options ONLY:\n"
    prompt += ", ".join(f"'{a}'" for a in VALID_ACUITY)
    prompt += "\n\nRespond with ONLY ONE of these five options. No explanation."
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
                max_tokens=100,    # one label — no more needed
                extra_body={"provider": {"zdr": True}},  # Zero Data Retention
            )

            raw = response.choices[0].message.content
            if raw is None:
                return "Prediction failed"

            raw = raw.strip()
            for acuity in VALID_ACUITY:
                if acuity.lower() in raw.lower():
                    return acuity

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

    remaining = df[~df['encounterkey'].astype(str).isin(processed_ids)]
    total     = len(remaining)
    print(f"[{condition}] {total} records remaining | {N_WORKERS} workers")

    batch_size    = 50
    total_batches = (total + batch_size - 1) // batch_size
    start_time    = time.time()
    processed_count = 0

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
            row_copy['condition']        = condition
            row_copy['predicted_acuity'] = batch_results.get(ek, "Prediction failed")
            rows.append(row_copy)

        processed_df  = pd.concat([processed_df, pd.DataFrame(rows)], ignore_index=True)
        processed_ids.update(str(r['encounterkey']) for r in rows)
        processed_count += len(rows)
        processed_df.to_csv(checkpoint_file, index=False)

        elapsed = time.time() - start_time
        rps     = processed_count / elapsed
        eta     = (total - processed_count) / rps if rps > 0 else float('inf')
        print(f"[{condition}] Progress: {processed_count}/{total} | {rps:.3f} rec/s | ETA: {eta/60:.1f} min")

    return processed_df


# ── Accuracy ──────────────────────────────────────────────────────────────────
def calculate_accuracy(df, true_col='acuitylevel'):
    for condition, grp in df.groupby('condition'):
        valid = grp[grp['predicted_acuity'].isin(VALID_ACUITY)].copy()
        invalid = len(grp) - len(valid)
        if invalid > 0:
            print(f"[{condition}] Warning: {invalid} rows excluded")
        matches  = (valid['predicted_acuity'] == valid[true_col]).sum()
        total    = len(valid)
        accuracy = matches / total if total > 0 else 0
        print(f"\n[{condition}] Overall Accuracy: {matches}/{total} = {accuracy:.2%}")
        for level in VALID_ACUITY:
            lvl = valid[valid[true_col] == level]
            if len(lvl) > 0:
                acc = (lvl['predicted_acuity'] == lvl[true_col]).sum() / len(lvl)
                print(f"  {level:<12}: {acc:.2%} ({len(lvl)} records)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load your dataset — must contain: encounterkey, primarychiefcomplaintname,
    # and optionally: Age, sex, firstrace, acuitylevel (for evaluation)
    # df = pd.read_csv("your_data.csv")

    # Extract vital signs from ED provider notes (required preprocessing step)
    df['Vital_Signs'] = df['ED_Provider_Notes_Text'].apply(extract_vital_signs)
    extracted = (df['Vital_Signs'] != "No vital signs available").sum()
    print(f"Vital signs extracted: {extracted}/{len(df)} rows ({extracted/len(df):.1%})")
    print(f"Loaded {len(df)} records\n")

    # ── Zero-shot ─────────────────────────────────────────────────────────────
    zs_df = run_condition(
        df,
        condition="zero_shot",
        prompt_fn=build_zero_shot_prompt,
        checkpoint_file="acuity_zero_shot_checkpoint.csv",
    )

    # ── Step-back ─────────────────────────────────────────────────────────────
    principles = get_stepback_principles()
    sb_df = run_condition(
        df,
        condition="step_back",
        prompt_fn=lambda row: build_stepback_prompt(row, principles),
        checkpoint_file="acuity_step_back_checkpoint.csv",
    )

    # ── Combine and save ──────────────────────────────────────────────────────
    results_df = pd.concat([zs_df, sb_df], ignore_index=True)
    results_df.to_csv("acuity_results.csv", index=False)
    print("\nSaved to acuity_results.csv")

    calculate_accuracy(results_df, true_col='acuitylevel')