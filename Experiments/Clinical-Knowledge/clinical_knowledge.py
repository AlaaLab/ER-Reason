import os
import time
import random
import threading
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
CHECKPOINT_FILE = "clinical_knowledge_checkpoint.csv"
OUTPUT_FILE     = "clinical_knowledge_results.csv"

checkpoint_lock = threading.Lock()

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
    
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an experienced emergency medicine physician.
You will be given a patient presentation and a list of diagnoses to consider, along with supporting clinical evidence for each.
Your task is to identify the single most likely diagnosis exactly how it is written from the provided list.
Respond with ONLY the diagnosis name. Do not include any explanation, punctuation, or additional text."""


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_encounter_prompt(row):
    """
    Give the model the patient summary, then all (differential, evidence) pairs,
    and ask for the single most likely diagnosis from the fixed list.
    """
    one_sentence = row['One_Sentence_Extracted']

    steps = []
    for t in range(1, 6):
        dx = row.get(f'differential_{t}')
        ev = row.get(f'evidence_{t}')
        if pd.notna(dx) and pd.notna(ev) and str(dx).strip() != '' and str(ev).strip() != '':
            steps.append((str(dx).strip().rstrip(':').strip(), str(ev).strip()))

    if not steps:
        return None, []

    all_dxs = [dx for dx, _ in steps]

    prompt  = f"=== PATIENT CASE ===\n{one_sentence}\n"
    prompt += f"\n=== DIAGNOSES TO CONSIDER ===\n"
    for dx in all_dxs:
        prompt += f"- {dx}\n"

    prompt += f"\n=== CLINICAL EVIDENCE ===\n"
    for i, (dx, ev) in enumerate(steps, 1):
        prompt += f"Evidence {i} (related to {dx}): {ev}\n"

    prompt += f"\n=== TASK ===\n"
    prompt += f"Based on the patient case and all clinical evidence above, what is the single most likely diagnosis?\n"
    prompt += f"You MUST choose exactly one diagnosis from the list above. Respond with ONLY the diagnosis name, nothing else."

    return prompt, all_dxs


# ── API call with retry ───────────────────────────────────────────────────────
def call_model(prompt, all_dxs, retries=MAX_RETRIES):
    pool_lower = {dx.lower().strip() for dx in all_dxs}
    backoff    = 2

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,  # set to 0 for non-reasoning models; some reasoning
                                  # models (o4-mini) do not support this parameter —
                                  # remove the line if you hit a 400 error.
                max_tokens=3500,  # sufficient for a diagnosis name; reasoning models
                extra_body={"provider": {"zdr": True}},  # Zero Data Retention — only routes to ZDR-compliant providers
                                  # consume tokens internally before emitting the answer —
                                  # increase to 10000 if you see empty responses.
            )
            answer = response.choices[0].message.content.strip().rstrip(':').strip()

            # Exact match (case-insensitive)
            if answer.lower().strip() in pool_lower:
                for dx in all_dxs:
                    if dx.lower().strip() == answer.lower().strip():
                        return dx, None

            # Partial match fallback
            for dx in all_dxs:
                if dx.lower().strip() in answer.lower() or answer.lower() in dx.lower().strip():
                    return dx, None

            print(f"  Answer not in pool: '{answer}' | pool: {all_dxs} (attempt {attempt + 1})")
            time.sleep(backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 30)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = backoff + random.uniform(0, 2)
                print(f"  Rate limit, waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
            else:
                print(f"  Error (attempt {attempt + 1}): {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    return None, "max_retries_exceeded"


# ── Per-encounter worker ──────────────────────────────────────────────────────
def process_encounter(idx, row):
    ek            = row['encounterkey']
    prompt, all_dxs = build_encounter_prompt(row)

    if prompt is None:
        return {
            'encounterkey':    ek,
            'all_dxs':         '[]',
            'final_diagnosis': None,
            'parse_error':     'no_populated_steps',
        }

    prediction, error = call_model(prompt, all_dxs)

    return {
        'encounterkey':    ek,
        'all_dxs':         str(all_dxs),
        'final_diagnosis': prediction,
        'parse_error':     error,
    }


# ── Resume from checkpoint ────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = pd.read_csv(CHECKPOINT_FILE)
    done_eks   = set(checkpoint['encounterkey'].tolist())
    results    = checkpoint.to_dict('records')
    print(f"Resuming from checkpoint: {len(done_eks)} encounters done")
else:
    done_eks = set()
    results  = []

# ── Load and prep data ────────────────────────────────────────────────────────
# `test` should be a DataFrame already loaded and filtered to SCT encounterkeys.
# Required columns: encounterkey, One_Sentence_Extracted,
#                   differential_1..5, evidence_1..5

keep_cols = [
    'encounterkey', 'One_Sentence_Extracted',
    'differential_1', 'evidence_1',
    'differential_2', 'evidence_2',
    'differential_3', 'evidence_3',
    'differential_4', 'evidence_4',
    'differential_5', 'evidence_5',
]
enc_df = test[keep_cols].drop_duplicates('encounterkey').reset_index(drop=True)
print(f"Total encounters: {len(enc_df)}")

remaining = [(idx, row) for idx, row in enc_df.iterrows()
             if row['encounterkey'] not in done_eks]
print(f"Remaining:        {len(remaining)}")

# ── Run ───────────────────────────────────────────────────────────────────────
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_encounter, idx, row): idx
               for idx, row in remaining}

    with tqdm(total=len(remaining), desc="Encounters") as pbar:
        for future in as_completed(futures):
            result = future.result()
            with checkpoint_lock:
                results.append(result)
                pbar.update(1)
                if len(results) % 25 == 0:
                    pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)

# ── Save final ────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDone.")
print(f"Total encounters:  {len(results_df)}")
print(f"Valid predictions: {results_df['final_diagnosis'].notna().sum()}")
print(f"Errors:            {results_df['parse_error'].notna().sum()}")
print(results_df['parse_error'].value_counts(dropna=False))