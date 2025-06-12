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
  - 
## 🚀 Getting Started

### **Step 1: Clone the ER-Reason repository**

```bash
git clone https://github.com/AlaaLab/ER-Reason.git
```

### **Step 2: Install dependencies**

```bash
pip install -r <path-to-repo>/ER-Reason/requirements.txt
```

### **Step 3: Download the dataset**

Download the ER-Reason dataset from PhysioNet:  
https://physionet.org/projects/JGAP8qn2p4CPnPeXceVE/overview/

> **Note:** Access requires registration and data use agreement approval.




### ER-Reason Column Descriptions

| Column Name | Description |
|------------|-------------|
| `patientdurablekey` | Unique patient identifier |
| `encounterkey` | Unique encounter identifier associated with the current visit to the ER |
| `primarychiefcomplaintname` | Chief complaint when the patient came into the ER |
| `primaryeddiagnosisname` | Diagnosis given from the ER doctor at the end of current ER visit |
| `sex` | Patient's sex |
| `firstrace` | Patient's race |
| `preferredlanguage` | Patient's preferred language |
| `highestlevelofeducation` | Patient's highest level of education |
| `maritalstatus` | Patient's marital status |
| `Age` | Patient's age |
| `Discharge_Summary_Note_Key` | Unique identifier linking to the historical discharge summary note |
| `Progress_Note_Key` | Unique identifier linking to the historical progress note |
| `HP_Note_Key` | Unique identifier linking to the historical history and physical note |
| `Echo_Key` | Unique identifier linking to the historical echo note |
| `Imaging_Key` | Unique identifier linking to the historical imaging note |
| `Consult_Key` | Unique identifier linking to the historical consult note |
| `ED_Provider_Notes_Key` | Unique identifier for the current visit's ED provider note |
| `ECG_Key` | Unique identifier linking to the historical ECG note |
| `Discharge_Summary_Text` | **Historical**: Discharge summary text from patient's previous hospital encounter |
| `Progress_Note_Text` | **Historical**: Progress note text from patient's previous hospital encounter |
| `HP_Note_Text` | **Historical**: History and physical note from patient's previous hospital encounter |
| `Echo_Text` | **Historical**: Echocardiogram results and interpretation from patient's previous hospital encounter |
| `Imaging_Text` | **Historical**: Imaging reports and findings from patient's previous hospital encounter |
| `Consult_Text` | **Historical**: Specialist consultation notes from patient's previous hospital encounter |
| `ECG_Text` | **Historical**: Electrocardiogram results and interpretation from patient's previous hospital encounter |
| `ED_Provider_Notes_Text` | **Current Visit**: ED Provider note from the current ER visit (associated with this encounter, patient, chief complaint, and diagnosis) |
| `One_Sentence_Extracted` | Key one-liner summary extracted from the current ED provider note |
| `note_count` | Number of notes associated with the patient in this dataset (minimum 2: ED and discharge summary, increases based on availability) |
| `acuitylevel` | Assigned ESI (Emergency Severity Index) level at triage when patient arrived at ER |
| `eddisposition` | Assigned disposition when patient left ER (e.g., discharged, admitted, transferred) |
| `ArrivalYearKey` | Year patient arrived at the ER for current visit |
| `DepartureYearKeyValue` | Year patient departed from the ER for current visit |
| `DepartureYearKey` | Year patient departed from the ER (key format) |
| `DispositionYearKeyValue` | Year the disposition was assigned |
| `birthYear` | Year when patient was born |
| `Discharge_Summary_Year` | Year the historical discharge summary was created |
| `Progress_Note_Year` | Year the historical progress note was created |
| `HP_Note_Year` | Year the historical history and physical note was created |
| `Echo_Year` | Year the historical echo was performed |
| `Imaging_Year` | Year the historical imaging was performed |
| `Consult_Year` | Year the historical consult was completed |
| `ED_Provider_Notes_Year` | Year the current ED provider notes were created |
| `ECG_Year` | Year the historical ECG was performed |
| `Rule_Out` | Differential diagnosis list made by the physician given the chief complaint, demographics, and one-liner (acts as a mental model for the pre-encounter) |
| `Decision_Factors` | Factors doctors would deploy to narrow down their differential list |
| `Treatment_Plan` | Factors and treatment plan the physician would choose given the history and physical |

