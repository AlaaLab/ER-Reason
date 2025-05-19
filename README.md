# ER-Reason: A Benchmark Dataset for LLM-Based Clinical Reasoning in the Emergency Room

## What is ER-Reason?

ER-Reason is a large-scale benchmark suite for evaluating the clinical reasoning capabilities of large language models (LLMs) in the emergency room (ER) — a high-stakes environment where clinicians make rapid, life-critical decisions.

ER-Reason is designed to move beyond multiple-choice exam-style QA and instead test LLMs on realistic, multi-stage clinical workflows grounded in real-world electronic health records (EHRs). ER-Reason simulates the full ER decision-making pipeline—including triage, treatment selection, and final diagnosis—and includes expert-written rationales to capture the step-by-step thinking used by physicians in real clinical settings.


## ✅ Key Features

- **📚 Real-World Clinical Data**  
  Includes **3,984 patients** and **25,174 de-identified longitudinal clinical notes** from an academic medical center. Document types include:
  - Discharge summaries
  - Progress notes
  - History & Physical (H&P)
  - Consult notes
  - Imaging reports
  - Echocardiography reports
  - ER provider notes

- **🧠 Clinical Reasoning Annotations**  
  72 physician-authored rationales explaining the reasoning behind clinical decisions—modeled after residency-level teaching and not typically found in EHR documentation.

- **⚕️ Workflow-Aligned Clinical Tasks**  
  Tasks are structured around the actual ER care process:
  - Triage Intake
  - EHR Review
  - Initial Assessment
  - Treatment Planning
  - Disposition Planning (Admit, Discharge, ICU)
  - Final Diagnosis

- **🧠 Clinical Reasoning Annotations**  
  72 physician-authored rationales explaining the reasoning behind clinical decisions—modeled after residency-level teaching and not typically found in EHR documentation.

- **🤖 Model Compatibility**  
  ER-Reason includes code and templates for evaluating:
  - **LLaMA 3.2-3B-Instruct**
  - **GPT-4o**
  - **GPT-3.5 Turbo**
  - **O3-Mini**

- **🧪 LLM Evaluation Tasks**  
  Benchmark includes structured tasks and metrics to evaluate:
  - Acuity classification: determining patient urgency based on symptoms and clinical history
  - EHR summarization: summarizing key aspects of a patient's clinical history
  - Rationale generation: creating the reasoning behind clinical decisions, aligned with physician thinking
  - Diagnosis inference: inferring the most likely diagnosis based on EHR data and symptoms
  - Disposition prediction: predicting whether a patient should be admitted, discharged, or sent to the ICU.

---
# GitHub Repo Includes Code for Tasks across: Acuity, Disposition, Diagnosis, EHR Review Summarization, and Rationale that can be applied for the following models:
1. Llama 3.0 
2. GPT-4o
3. GPT 3.5-Turbo
4. O3-Mini  
