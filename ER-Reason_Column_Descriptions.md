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
| `Discharge_Summary_Text` | Discharge summary text from patient's previous hospital encounter |
| `Progress_Note_Text` |  Progress note text from patient's previous hospital encounter |
| `HP_Note_Text` | History and physical note from patient's previous hospital encounter |
| `Echo_Text` | Echocardiogram results and interpretation from patient's previous hospital encounter |
| `Imaging_Text` | Imaging reports and findings from patient's previous hospital encounter |
| `Consult_Text` | Specialist consultation notes from patient's previous hospital encounter |
| `ECG_Text` | Electrocardiogram results and interpretation from patient's previous hospital encounter |
| `ED_Provider_Notes_Text` | Current Visit: ED Provider note from the current ER visit (associated with this encounter, patient, chief complaint, and diagnosis) |
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

