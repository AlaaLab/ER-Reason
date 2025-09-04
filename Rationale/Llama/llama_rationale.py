from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
import re

# Load your local LLaMA 3.2-3B model
model_path = '...Llama-3.2-3B-Instruct/'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LLaMA inference function
def query_llama(prompt, max_new_tokens=1200, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def predict_ror_with_dynamic_notes(row):
    # Extract all available fields
    chief_complaint = row['primarychiefcomplaintname']
    sex = row['sex']
    age = row['Age']
    one_liner = row['One_Sentence_Extracted']
    ed_presentation = row['ED_Presentations']

    # Check if necessary basic fields exist
    if pd.isna(chief_complaint):
        return None, None, None, None

    # Initialize available note types with their existence status
    available_notes = {
        'Discharge Summary': not pd.isna(row.get('Discharge_Summary_Text')),
        'Progress Notes': not pd.isna(row.get('Progress_Note_Text')),
        'H&P': not pd.isna(row.get('HP_Note_Text')),
        'Echo': not pd.isna(row.get('Echo_Text')),
        'Imaging': not pd.isna(row.get('Imaging_Text')),
        'Consult': not pd.isna(row.get('Consult_Text')),
        'ECG': not pd.isna(row.get('ECG_Text')),
    }

    # Step 1: Ask the model which notes it wants to see (always include Discharge Summary if available)
    url = f"{RESOURCE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    # First message to decide which notes to read
    selection_payload = {
        "messages": [
            {"role": "user", "content": "You are an experienced Emergency Department (ED) physician. Your task is to decide which medical notes you need to have a clinical understanding of before you see the patient and create your differential list."},
            {"role": "user", "content": f"Patient basic info: {age}yo {sex} with chief complaint: {chief_complaint}\n\n"
                                      f"Available notes (respond ONLY with the names of notes you want to see, separated by commas):\n"
                                      f"- Discharge Summary: {'Available' if available_notes['Discharge Summary'] else 'Not available'}\n"
                                      f"- Progress Notes: {'Available' if available_notes['Progress Notes'] else 'Not available'}\n"
                                      f"- H&P: {'Available' if available_notes['H&P'] else 'Not available'}\n"
                                      f"- Echo: {'Available' if available_notes['Echo'] else 'Not available'}\n"
                                      f"- Imaging: {'Available' if available_notes['Imaging'] else 'Not available'}\n"
                                      f"- Consult: {'Available' if available_notes['Consult'] else 'Not available'}\n"
                                      f"- ECG: {'Available' if available_notes['ECG'] else 'Not available'}\n"
                                      f"Based on the chief complaint, list ONLY the note types you need to review (comma-separated, no explanation). Always include Discharge Summary if available:"}
        ],
        "max_completion_tokens": 1000
    }

    # Request note selection
    retries = 0
    requested_notes = []
    notes_requested_str = ""  # String to track requested notes

    while retries < 3:
        try:
            selection_response = requests.post(url, headers=headers, json=selection_payload)
            selection_response.raise_for_status()

            # Get requested note types
            notes_text = selection_response.json()["choices"][0]["message"]["content"].strip()
            requested_notes = [note.strip() for note in notes_text.split(',')]

            # Save the original request for the tracking column
            notes_requested_str = notes_text

            # Filter out unavailable notes
            requested_notes = [note for note in requested_notes
                              if note in available_notes.keys() and available_notes[note]]

            # Always include Discharge Summary if available and not already requested
            if available_notes['Discharge Summary'] and 'Discharge Summary' not in requested_notes:
                requested_notes.append('Discharge Summary')
                if notes_requested_str:
                    notes_requested_str += ", Discharge Summary (auto-added)"
                else:
                    notes_requested_str = "Discharge Summary (auto-added)"

            break
        except requests.exceptions.RequestException as e:
            print(f"Note selection request failed: {e}. Retrying {retries+1}/3...")
            time.sleep(5)
            retries += 1

    if not requested_notes and available_notes['Discharge Summary']:
        # Default to discharge summary if selection failed but it's available
        requested_notes = ['Discharge Summary']
        notes_requested_str = "Failed to get selections. Defaulted to: Discharge Summary"
    elif not requested_notes:
        # If selection failed and no discharge summary, use whatever is available
        available_notes_list = [note for note, available in available_notes.items() if available][:2]
        requested_notes = available_notes_list
        notes_requested_str = f"Failed to get selections. Defaulted to: {', '.join(available_notes_list)}"

    # Step 2: Prepare the actual notes content
    notes_content = ""

    # Map note types to DataFrame column names
    note_type_to_column = {
        'Discharge Summary': 'Discharge_Summary_Text',
        'Progress Notes': 'Progress_Note_Text',
        'H&P': 'HP_Note_Text',
        'Echo': 'Echo_Text',
        'Imaging': 'Imaging_Text',
        'Consult': 'Consult_Text',
        'ECG': 'ECG_Text',
    }
    
    # Function to truncate text to manage token limits
    def truncate_text(text, max_chars=3000):
        if text and len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    # Add requested notes to content
    for note_type in requested_notes:
        column_name = note_type_to_column.get(note_type)
        if column_name and not pd.isna(row.get(column_name)):
            notes_content += f"\n\n{note_type}:\n{truncate_text(row[column_name])}"

    # ==================== MODIFIED CODE STARTS HERE ====================
    # Instead of making one API call, we'll make three separate calls for each section
    
    # Common system message for context
    system_message = "You are an experienced Emergency Department (ED) physician tasked with creating a comprehensive assessment and plan for patients. Based on the available clinical information, you will provide: 1) A differential diagnosis list, 2) Medical decision factors including necessary exams/imaging to rule out diagnoses, and 3) A treatment plan including disposition and recommendations. Answer these questions:\n"

    # 1. First API call - DIFFERENTIAL DIAGNOSIS (without ED presentation)
    differential_system_message = system_message + " DIFFERENTIAL QUESTION: What are your initial differential diagnoses based on the past medical history (PREVIOUS HOSPITAL ENCOUNTER) and the chief complaint(CURRENT CHIEF COMPLAINT -- THIS IS THE TREATMENT FOCUS), and why? Please also address which diagnoses are lowest on your differential and why"
    
    differential_payload = {
        "messages": [
            {"role": "user", "content": f"{differential_system_message}\n"
                                       f"Based on the patient's chief complaint, age, sex, and available clinical information, provide:\n\n"
                                       f"DIFFERENTIAL DIAGNOSIS: List of differential diagnoses to consider for the CURRENT chief complaint given clinically relevant information such as the PAST medical notes.\n\n"
                                       f"IMPORTANT: Everything in the notes is from PAST encounters. The patient is NOW presenting with a NEW complaint: Chief Complaint: {chief_complaint}\n"
                                       f"Age: {age}\n"
                                       f"Sex: {sex}\n"
                                       f"Patient One-liner Summary: {one_liner}\n"
                                       f"REMINDER: All of these notes are from the PREVIOUS hospital encounter for this patient. This should only give you context for their current complaint:{notes_content}"}
        ],
        "max_completion_tokens": 6000
    }

    # 2. Second API call - MEDICAL DECISION FACTORS (without ED presentation)
    decision_system_message = system_message + " DECISION FACTORS: What additional information or studies would you want in order to help narrow your differential diagnoses? How would you weigh their relative importance? How would you weigh their relative importance?"
    
    decision_payload = {
        "messages": [
            {"role": "user", "content": f"{decision_system_message}\n"
                                      f"Based on the patient's chief complaint, age, sex, and available clinical information, provide:\n\n"
                                      f"MEDICAL DECISION FACTORS: List what important additional information you'd like to know on history, what important exam findings you'd assess for, and what imaging/tests you would do, in addition to any formal consultations you'd made with specialists -- this should be items that would be necessary to rule out diagnoses on the differential list for the CURRENT chief complaint. NOTE: NOTHING has currently been performed.\n\n"
                                      f"IMPORTANT: Everything in the notes is from PAST encounters. The patient is NOW presenting with a NEW complaint: Chief Complaint: {chief_complaint}\n"
                                      f"Age: {age}\n"
                                      f"Sex: {sex}\n"
                                      f"Patient One-liner Summary: {one_liner}\n"
                                      f"REMINDER: All of these notes are from the PREVIOUS hospital encounter for this patient. This should only give you context for their current complaint:{notes_content}"}
        ],
        "max_completion_tokens": 6000
    }

    # 3. Third API call - TREATMENT PLAN (WITH ED presentation)
    treatment_system_message = system_message + " TREATMENT PLAN: Given the additional history and physical exam provided in the ED provider note, which additional diagnostic tests would you like to order and what is your recommended initial treatment plan?"
    
    treatment_payload = {
        "messages": [
            {"role": "user", "content": f"{treatment_system_message}\n"
                                      f"Based on the patient's chief complaint, age, sex, and available clinical information, provide:\n\n"
                                      f"TREATMENT PLAN: Given the current history and physical provided in the ED note, what do you think is the most likely diagnosis and what is your recommended treatment plan? If you do not have enough information to conclude a final diagnosis, describe what information you still need. Include disposition (discharge, admit, ICU, etc.) and treatment recommendations for the patient given the history and physical exam in the current ER visit.\n\n"
                                      f"IMPORTANT: Everything in the notes is from PAST encounters. The patient is NOW presenting with a NEW complaint: Chief Complaint: {chief_complaint}\n"
                                      f"Age: {age}\n"
                                      f"Sex: {sex}\n"
                                      f"Patient One-liner Summary: {one_liner}\n"
                                      f"REMINDER: All of these notes are from the PREVIOUS hospital encounter for this patient. This should only give you context for their current complaint:{notes_content}"}
        ],
        "max_completion_tokens": 6000
    }

    # Add ED Presentation ONLY to the treatment plan payload
    if pd.notna(row['ED_Presentations']):
        treatment_payload["messages"][0]["content"] += f"\nCURRENT ED Encounter Physical associated with the CURRENT CHIEF COMPLAINT: {row['ED_Presentations']}"

    # Send the three separate API requests
    responses = {}

    try:
        # Call 1: Differential Diagnosis
        response_differential = requests.post(url, headers=headers, json=differential_payload)
        responses['Differential Diagnosis'] = response_differential.json()

        # Call 2: Medical Decision Factors
        response_decision = requests.post(url, headers=headers, json=decision_payload)
        responses['Medical Decision Factors'] = response_decision.json()

        # Call 3: Treatment Plan
        response_treatment = requests.post(url, headers=headers, json=treatment_payload)
        responses['Treatment Plan'] = response_treatment.json()

    except requests.exceptions.RequestException as e:
        print(f"API requests failed: {e}")

    return responses, notes_requested_str


# Apply function to DataFrame - now returns four values
results = df.apply(lambda row: predict_ror_with_dynamic_notes(row), axis=1)

# Split the results into four columns, handling potential None values
df["LLM_Differential_Diagnosis"] = [result[0] if result[0] is not None else "No response" for result in results]
df["LLM_Medical_Decision_Factors"] = [result[1] if result[1] is not None else "No response" for result in results]
df["LLM_Treatment_Plan"] = [result[2] if result[2] is not None else "No response" for result in results]
df["Requested_Notes"] = [result[3] if result[3] is not None else "No response" for result in results]

# Preview the DataFrame to confirm
print(df.head())
