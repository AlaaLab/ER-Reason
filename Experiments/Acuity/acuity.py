import os
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
CHECKPOINT_FILE = "acuity_checkpoint.csv"
OUTPUT_FILE     = "acuity_results.csv"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are an experienced Emergency Department triage nurse."

VALID_ACUITY = ['Immediate', 'Emergent', 'Urgent', 'Less Urgent', 'Non-Urgent']


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(row):
    if 'primarychiefcomplaintname' not in row or pd.isna(row['primarychiefcomplaintname']):
        return None

    prompt = "Predict the emergency department acuity level for this patient.\n\n"

    if 'Age' in row and not pd.isna(row['Age']):
        prompt += f"Age: {row['Age']}\n"
    if 'sex' in row and not pd.isna(row['sex']):
        prompt += f"Sex: {row['sex']}\n"
    if 'firstrace' in row and not pd.isna(row['firstrace']):
        prompt += f"Race: {row['firstrace']}\n"

    prompt += f"Chief Complaint: {row['primarychiefcomplaintname']}\n"

    if 'Vital_Signs' in row and not pd.isna(row['Vital_Signs']):
        prompt += f"Vital Signs: {row['Vital_Signs']}\n"

    prompt += "\nSelect the most appropriate acuity level from the following options ONLY:\n"
    prompt += ", ".join(f"'{a}'" for a in VALID_ACUITY)
    prompt += "\n\nRespond with ONLY ONE of these five options. No explanation."

    return prompt


# ── API call with retry ───────────────────────────────────────────────────────
def predict_acuity(row, max_retries=MAX_RETRIES):
    prompt       = build_prompt(row)
    encounterkey = str(row['encounterkey']) if 'encounterkey' in row else None

    if prompt is None:
        return encounterkey, None

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
                return encounterkey, "Prediction failed"

            raw = raw.strip()
            for acuity in VALID_ACUITY:
                if acuity.lower() in raw.lower():
                    return encounterkey, acuity

            print(f"Unmatched response for {encounterkey}: '{raw[:100]}'")
            return encounterkey, raw

        except Exception as e:
            err = str(e)
            if '429' in err or 'rate_limit' in err.lower() or 'Connection' in err:
                wait = backoff + random.uniform(0, 1)
                print(f"Rate limit / connection error. Waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
            else:
                print(f"Error on {encounterkey}: {e}")
                return encounterkey, "Prediction failed"

    return encounterkey, "Prediction failed"


# ── Checkpointed parallel runner ──────────────────────────────────────────────
def process_with_checkpoints(df, batch_size=50, checkpoint_file=CHECKPOINT_FILE, max_workers=N_WORKERS):
    if os.path.exists(checkpoint_file):
        processed_df  = pd.read_csv(checkpoint_file)
        processed_ids = set(processed_df['encounterkey'].astype(str).tolist())
        print(f"Resuming from checkpoint: {len(processed_ids)} already processed")
    else:
        processed_df  = pd.DataFrame(columns=df.columns.tolist() + ['predicted_acuity'])
        processed_ids = set()

    remaining     = df[~df['encounterkey'].astype(str).isin(processed_ids)]
    total         = len(remaining)
    total_batches = (total + batch_size - 1) // batch_size
    start_time    = time.time()
    processed_count = 0

    print(f"{total} records remaining | {max_workers} parallel workers")

    for batch_num in range(total_batches):
        batch = remaining.iloc[batch_num * batch_size:(batch_num + 1) * batch_size]
        print(f"\nBatch {batch_num+1}/{total_batches} ({len(batch)} records)")

        batch_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(predict_acuity, row): idx
                       for idx, row in batch.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures)):
                encounterkey, prediction = future.result()
                batch_results[encounterkey] = prediction

        for idx, row in batch.iterrows():
            ek          = str(row['encounterkey'])
            row_copy    = row.copy()
            row_copy['predicted_acuity'] = batch_results.get(ek, "Prediction failed")
            processed_df  = pd.concat([processed_df, pd.DataFrame([row_copy])], ignore_index=True)
            processed_ids.add(ek)
            processed_count += 1

        processed_df.to_csv(checkpoint_file, index=False)

        elapsed = time.time() - start_time
        rps     = processed_count / elapsed
        eta     = (total - processed_count) / rps if rps > 0 else float('inf')
        print(f"Progress: {processed_count}/{total} | {rps:.3f} rec/s | ETA: {eta/60:.1f} min")

    return processed_df


# ── Accuracy ──────────────────────────────────────────────────────────────────
def calculate_accuracy(df, true_col='acuitylevel'):
    valid_df = df[df['predicted_acuity'].isin(VALID_ACUITY)].copy()
    invalid  = len(df) - len(valid_df)
    if invalid > 0:
        print(f"Warning: {invalid} rows with invalid/missing predictions excluded")

    matches  = (valid_df['predicted_acuity'] == valid_df[true_col]).sum()
    total    = len(valid_df)
    accuracy = matches / total if total > 0 else 0
    print(f"\nOverall Accuracy: {matches}/{total} = {accuracy:.2%}")

    print("\nAccuracy by acuity level:")
    for level in VALID_ACUITY:
        lvl_df = valid_df[valid_df[true_col] == level]
        if len(lvl_df) > 0:
            lvl_acc = (lvl_df['predicted_acuity'] == lvl_df[true_col]).sum() / len(lvl_df)
            print(f"  {level:<12}: {lvl_acc:.2%} ({len(lvl_df)} records)")

    return accuracy


# ── Vital signs extraction ────────────────────────────────────────────────────
# Vital_Signs is not a standard column in the ER-Reason dataset.
# Run this function on ED_Provider_Notes_Text to extract it before running
# the experiment. The output is stored in a new column called Vital_Signs.

import re

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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load your dataset — must contain: encounterkey, primarychiefcomplaintname,
    # and optionally: Age, sex, firstrace, acuitylevel (for evaluation)
    # df = pd.read_csv("your_data.csv")

    # Extract vital signs from ED provider notes (required preprocessing step)
    df['Vital_Signs'] = df['ED_Provider_Notes_Text'].apply(extract_vital_signs)
    extracted = (df['Vital_Signs'] != "No vital signs available").sum()
    print(f"Vital signs extracted: {extracted}/{len(df)} rows ({extracted/len(df):.1%})")

    print(f"Loaded {len(df)} records")

    results_df = process_with_checkpoints(df)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    calculate_accuracy(results_df, true_col='acuitylevel')