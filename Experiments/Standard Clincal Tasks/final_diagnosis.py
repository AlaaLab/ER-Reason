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

MAX_RETRIES     = 5
N_WORKERS       = 10
PRINCIPLES_FILE = "diagnosis_principles.txt"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an experienced Emergency Department physician. "
    "Respond with only the ICD-10 code and diagnosis name in the format: "
    "[CODE] [DIAGNOSIS NAME]. Nothing else."
)


# ── API call with retry ───────────────────────────────────────────────────────
def call_model(prompt, encounterkey, max_retries=MAX_RETRIES):
# Truncates ED note before "Final Disposition and ED Course" and redacts the
# diagnosis name to prevent label leakage.

# ── Note cleaning ─────────────────────────────────────────────────────────────
# Truncates ED note before "Final Disposition and ED Course" and redacts the
# diagnosis name to prevent label leakage.

def clean_ed_note(row):
    note      = row.get('ED_Provider_Notes_Text', '')
    diagnosis = row.get('primaryeddiagnosisname', '')

    if pd.isna(note) or note == '':
        return None

    note  = str(note)
    match = re.search(r'(?i)final\s+disposition\s+and\s+ed\s+course', note)
    if match:
        note = note[:match.start()].strip()
    else:
        print(f"Warning: 'Final Disposition and ED Course' not found for "
              f"encounterkey {row.get('encounterkey', 'unknown')}")

    if not pd.isna(diagnosis) and diagnosis != '':
        note = re.sub(re.escape(str(diagnosis)), '[DIAGNOSIS REDACTED]', note, flags=re.IGNORECASE)

    return note if note.strip() else None


# ── Step-back: Call 1 — fetch principles once ─────────────────────────────────
def get_stepback_principles(principles_file=PRINCIPLES_FILE):
    """Fetch ED diagnosis principles once and cache to disk for reuse."""
    if os.path.exists(principles_file):
        with open(principles_file) as f:
            principles = f.read().strip()
        print(f"Loaded cached principles from {principles_file}")
        return principles

    print("Fetching ED diagnosis principles (step-back Call 1)...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an experienced Emergency Department physician."},
            {"role": "user", "content": (
                "What are the key clinical principles and reasoning frameworks used to arrive at "
                "a primary ED diagnosis? Describe how you integrate chief complaint, past medical "
                "history, ED provider notes, and clinical findings to identify the most likely "
                "diagnosis. What patterns, red flags, and diagnostic reasoning strategies should "
                "guide selecting a primary ICD-10 diagnosis in the emergency department?"
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
    if 'Discharge_Summary_Text' in row and not pd.isna(row['Discharge_Summary_Text']):
        fields += f"\nPast Medical History (Most Recent Discharge Summary):\n{str(row['Discharge_Summary_Text'])}\n"
    cleaned_note = clean_ed_note(row)
    if cleaned_note:
        fields += f"\nED Provider Note:\n{cleaned_note}\n"
    return fields


def build_zero_shot_prompt(row):
    """Zero-shot: patient fields + ICD-10 format instruction."""
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt  = "Predict the most likely primary ED diagnosis for this patient.\n\n"
    prompt += build_patient_fields(row)
    prompt += "\nBased on the clinical information above, predict the most likely primary ED diagnosis for this patient."
    prompt += "\nRespond with ONLY the ICD-10 code followed by the diagnosis name in this exact format:"
    prompt += "\n[ICD-10 CODE] [DIAGNOSIS NAME]"
    prompt += "\nExample: J18.9 Pneumonia, unspecified"
    prompt += "\nNo explanation, no alternatives, no additional text."
    return prompt


def build_stepback_prompt(row, principles):
    """Step-back Call 2: inject retrieved principles then patient fields."""
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt  = "Using the following clinical principles:\n"
    prompt += f"{principles}\n\n"
    prompt += "Now predict the most likely primary ED diagnosis for this patient:\n\n"
    prompt += build_patient_fields(row)
    prompt += "\nBased on the clinical information above, predict the most likely primary ED diagnosis for this patient."
    prompt += "\nRespond with ONLY the ICD-10 code followed by the diagnosis name in this exact format:"
    prompt += "\n[ICD-10 CODE] [DIAGNOSIS NAME]"
    prompt += "\nExample: J18.9 Pneumonia, unspecified"
    prompt += "\nNo explanation, no alternatives, no additional text."
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
                max_tokens=100,   # ICD-10 code + diagnosis name; increase for reasoning models
                                  # if you see empty responses
                extra_body={"provider": {"zdr": True}},  # Zero Data Retention
            )

            raw = response.choices[0].message.content
            if raw is None:
                return "Prediction failed"

            return raw.strip()

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
            row_copy['condition']           = condition
            row_copy['predicted_diagnosis'] = batch_results.get(ek, "Prediction failed")
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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load your dataset — must contain: encounterkey, primarychiefcomplaintname,
    # ED_Provider_Notes_Text, and optionally: Age, sex, Discharge_Summary_Text,
    # primaryeddiagnosisname, primaryeddiagnosiscode (for evaluation)
    # df = pd.read_csv("your_data.csv")

    print(f"Loaded {len(df)} records\n")

    # ── Zero-shot ─────────────────────────────────────────────────────────────
    zs_df = run_condition(
        df,
        condition="zero_shot",
        prompt_fn=build_zero_shot_prompt,
        checkpoint_file="diagnosis_zero_shot_checkpoint.csv",
    )

    # ── Step-back ─────────────────────────────────────────────────────────────
    principles = get_stepback_principles()
    sb_df = run_condition(
        df,
        condition="step_back",
        prompt_fn=lambda row: build_stepback_prompt(row, principles),
        checkpoint_file="diagnosis_step_back_checkpoint.csv",
    )

    # ── Combine and save ──────────────────────────────────────────────────────
    results_df = pd.concat([zs_df, sb_df], ignore_index=True)
    results_df.to_csv("diagnosis_results.csv", index=False)
    print("\nSaved to diagnosis_results.csv")
    print(f"Total records: {len(results_df)}")
    print(f"Valid predictions: {(results_df['predicted_diagnosis'] != 'Prediction failed').sum()}")