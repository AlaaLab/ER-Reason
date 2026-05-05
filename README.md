# ER-Reason: A Benchmark Dataset for LLM Clinical Reasoning in the Emergency Room

**Paper:** https://arxiv.org/abs/2505.22919 | **Dataset:** [PhysioNet](https://physionet.org/content/er-reason/1.0.0/) 

---

## Overview

ER-Reason is a benchmark for evaluating large language models (LLMs) on clinical reasoning across key stages of the emergency room (ER) workflow. Unlike benchmarks based on medical licensing exams, ER-Reason evaluates not just what decisions models make, but how their reasoning evolves as clinical evidence accumulates.

ER-Reason consists of two components:

- **Longitudinal clinical notes** from 3,984 hospital encounters comprising 25,174 de-identified clinical notes across discharge summaries, progress notes, H&Ps, consult notes, imaging reports, and ER provider notes — supporting evaluation across triage intake, disposition planning, and final diagnosis.
- **SCT reasoning evaluation** comprising 194 physician-authored patient cases annotated by two ER physicians (2,555 total annotations), with three metrics — DxUpdate, DxTrajectory, and FinalDx — that measure sequential belief updating against physician consensus.

---

## Dataset Access

The ER-Reason dataset is hosted on PhysioNet and requires credentialed access:

1. Register at [physionet.org](https://physionet.org) and complete CITI training.
2. Request access at: https://physionet.org/content/er-reason/1.0.0/
3. Once approved, download the dataset files.

> **Note:** The dataset contains de-identified patient data and is governed by a PhysioNet data use agreement. Do not share or redistribute.

### Key dataset files

 
| File | Description | Key columns |
|---|---|---|
| `er_reason.csv` | Main dataset — one row per encounter, includes all clinical note text, demographic fields, acuity level, disposition, and primary ED diagnosis | `patientdurablekey`, `encounterkey`, `primarychiefcomplaintname`, `primaryeddiagnosisname`, `acuitylevel`, `eddisposition`, `ED_Provider_Notes_Text`, `Discharge_Summary_Text`, `One_Sentence_Extracted` |
| `icd_10_codes.csv` | Ground-truth ICD-10 codes per encounter — one row per code, multiple rows per encounter | `patientdurablekey`, `encounterkey`, `value` (ICD-10 code), `displaystring` (diagnosis name) |
| `annotated_sct.csv` | SCT evaluation cases derived from `er_reason.csv` — one row per encounter, includes the one-sentence patient summary and up to 5 differential/evidence pairs per case | `encounterkey`, `One_Sentence_Extracted`, `differential_1`–`differential_5`, `evidence_1`–`evidence_5` |
| `sct-annotations.csv` | Master list of the 194 SCT case encounterkeys — used to filter `er_reason.csv` to SCT encounters | `encounterkey` |
| `sct_cleaned_annotations.csv` | Physician rationales for each differential/evidence step — one row per (encounter, differential) pair | `encounterkey`, `differential`, `rationale` |
| `gt_clean.csv` | Ground-truth physician consensus scores for SCT evaluation — one row per (encounter, differential) pair | `encounterkey`, `differential`, `dxupdate` (ordinal score −2 to +2), `dxtrajectory` (ranked dict of all differentials) |

The CCSR reference file used for diagnosis evaluation must be downloaded separately from AHRQ:
https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp

---

## Repository Structure

```
ER-Reason/
├── Experiments/
│   ├── Standard Clinical Tasks/
│   │   ├── acuity.py               # Acuity prediction (zero-shot + step-back)
│   │   ├── disposition.py          # Disposition prediction (zero-shot + step-back)
│   │   ├── final_diagnosis.py      # Final diagnosis prediction (zero-shot + step-back)
│   │   ├── diag_evaluation.py      # ICD-10 exact match + CCSR accuracy
│   │   └── cross_stage_analysis.py # Cross-stage workflow accuracy (Table 5)
│   └── SCT Reasoning/
│       ├── clinical_knowledge.py   # Clinical knowledge baseline (Table 3 CK column)
│       ├── sct.py                  # SCT evaluation — baseline, single oracle, full oracle
│       └── sct_eval.py             # DxUpdate, DxTrajectory, FinalDx, coherence, Figure 3
├── ER-Reason-V1-Archive/           # Original codebase (archived)
├── ER-Reason_Column_Descriptions.md
├── README.md
└── requirements.txt
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/AlaaLab/ER-Reason.git
cd ER-Reason
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your OpenRouter API key**

All experiment scripts use [OpenRouter](https://openrouter.ai), so you can swap in any supported model with a single line change.

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

All scripts also enable Zero Data Retention (`"provider": {"zdr": True}`) by default, routing requests only to ZDR-compliant providers.

---

## Running Experiments

### Swapping models

Each script has a `MODEL_NAME` variable at the top. Replace it with any OpenRouter model string:

```python
MODEL_NAME = "openai/gpt-5.2-20251211"
# MODEL_NAME = "openai/o4-mini"          # remove temperature parameter for this model
# MODEL_NAME = "deepseek/deepseek-r1"
# MODEL_NAME = "google/gemini-2.5-flash"
# MODEL_NAME = "anthropic/claude-sonnet-4-5"
# MODEL_NAME = "microsoft/phi-4"
```

To enable Claude thinking mode, add to the API call:
```python
extra_body={"thinking": {"type": "enabled", "budget_tokens": 10000}}
```

---

### Standard Clinical Tasks

All three scripts run zero-shot and step-back conditions in a single pass, saving results to a CSV with a `condition` column.

**Acuity prediction**
```bash
python Experiments/Standard\ Clinical\ Tasks/acuity.py
# Output: acuity_results.csv
```

**Disposition prediction**
```bash
python Experiments/Standard\ Clinical\ Tasks/disposition.py
# Output: disposition_results.csv
```

**Final diagnosis prediction**
```bash
python Experiments/Standard\ Clinical\ Tasks/final_diagnosis.py
# Output: diagnosis_results.csv
```

**Diagnosis evaluation** (ICD-10 exact match + CCSR accuracy)
```bash
python Experiments/Standard\ Clinical\ Tasks/diag_evaluation.py
# Reads: diagnosis_results.csv, icd_10_codes.csv, DXCCSR_v2025-1.CSV
```

**Cross-stage workflow accuracy** (Table 5)
```bash
python Experiments/Standard\ Clinical\ Tasks/cross_stage_analysis.py
# Reads: acuity_results.csv, disposition_results.csv, diagnosis_results.csv,
#        icd_10_codes.csv, DXCCSR_v2025-1.CSV
```

---

### SCT Reasoning

**Clinical knowledge baseline**
```bash
python Experiments/SCT\ Reasoning/clinical_knowledge.py
# Output: clinical_knowledge_results.csv
```

**SCT evaluation** — runs all three conditions (baseline, single oracle, full oracle)
```bash
python Experiments/SCT\ Reasoning/sct.py
# Output: sct_results.csv
#         sct_baseline_checkpoint.csv
#         sct_single_oracle_checkpoint.csv
#         sct_full_oracle_checkpoint.csv
```

Each condition checkpoints independently — if interrupted, re-running will resume from where it left off.

**SCT metrics and figures** (Tables 2, 3, 4 and Figure 3)
```bash
python Experiments/SCT\ Reasoning/sct_eval.py
# Reads: sct_results.csv, gt_clean.csv
# Output: top1_by_timestep.pdf, top1_by_timestep.png
```

---

## Citation

If you use ER-Reason in your research, please cite:

```bibtex
@inproceedings{@article{mehandru2025er,
  title={Er-reason: A benchmark dataset for llm-based clinical reasoning in the emergency room},
  author={Mehandru, Nikita and Golchini, Niloufar and Bamman, David and Zack, Travis and Molina, Melanie F and Alaa, Ahmed},
  journal={arXiv preprint arXiv:2505.22919},
  year={2025}
}
```

---

## License

The code in this repository is released under the MIT License. The dataset is governed by the PhysioNet Data Use Agreement and may not be redistributed.
