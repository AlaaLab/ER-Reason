import sys
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
import gc


df = pd.read_csv('...disposition/results.csv')

# Path to the model
model_path = '...Llama-3.2-3B-Instruct/'

tokenizer = AutoTokenizer.from_pretrained(model_path)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    device_map="auto"
)

def format_messages_for_llama(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"<|system|>\n{message['content']}</s>\n"
        elif message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}</s>\n"
        elif message["role"] == "assistant":
            prompt += f"<|assistant|>\n{message['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt

def select_notes(record, llm_pipeline):
    chief_complaint = df.loc[record, 'primarychiefcomplaintname']
    sex = df.loc[record, 'sex'] if 'sex' in df.columns else "Unknown"
    age = df.loc[record, 'Age'] if 'Age' in df.columns else "Unknown"
    
    available_notes = {
        'Discharge Summary': not pd.isna(df.loc[record, 'Discharge_Summary_Text']),
        'Progress Notes': not pd.isna(df.loc[record, 'Progress_Note_Text']),
        'H&P': not pd.isna(df.loc[record, 'HP_Note_Text']),
        'Echo': not pd.isna(df.loc[record, 'Echo_Text']),
        'Imaging': not pd.isna(df.loc[record, 'Imaging_Text']),
        'Consult': not pd.isna(df.loc[record, 'Consult_Text']),
        'ECG': not pd.isna(df.loc[record, 'ECG_Text']),
    }
    
    selection_messages = [
        {
            "role": "system",
            "content": "You are an experienced Emergency Department (ED) physician. Your task is to decide which medical notes you need to read to predict the patient's ED disposition based on the chief complaint, PMH, physical exam findings, age, and sex."
        },
        {
            "role": "user",
            "content": f"Patient basic info: {age}yo {sex} with chief complaint: {chief_complaint}\n\n"
                      f"Available notes (respond ONLY with the names of notes you want to see, separated by commas):\n"
                      f"- Discharge Summary: {'Available' if available_notes['Discharge Summary'] else 'Not available'}\n"
                      f"- Progress Notes: {'Available' if available_notes['Progress Notes'] else 'Not available'}\n"
                      f"- H&P: {'Available' if available_notes['H&P'] else 'Not available'}\n"
                      f"- Echo: {'Available' if available_notes['Echo'] else 'Not available'}\n"
                      f"- Imaging: {'Available' if available_notes['Imaging'] else 'Not available'}\n"
                      f"- Consult: {'Available' if available_notes['Consult'] else 'Not available'}\n"
                      f"- ECG: {'Available' if available_notes['ECG'] else 'Not available'}\n"
                      f"Based on the chief complaint, list ONLY the note types you need to review (comma-separated, no explanation):"
        }
    ]

    formatted_prompt = format_messages_for_llama(selection_messages)
    outputs = llm_pipeline(formatted_prompt, max_new_tokens=100, do_sample=False, temperature=0.5)
    selection_text = outputs[0]["generated_text"]

    try:
        if "<|assistant|>" in selection_text:
            selection_text = selection_text.split("<|assistant|>")[-1].strip()
        selection_text = selection_text.replace("</s>", "").strip()
        requested_notes = [note.strip() for note in selection_text.split(',')]
        requested_notes = [note for note in requested_notes if note in available_notes and available_notes[note]]
    except Exception as e:
        print(f"Error parsing notes selection for record {record}: {e}")
        requested_notes = []

    if available_notes['Discharge Summary'] and 'Discharge Summary' not in requested_notes:
        requested_notes.append('Discharge Summary')
    if not requested_notes:
        requested_notes = [k for k, v in available_notes.items() if v][:2]

    return requested_notes, selection_text

def prepare_notes_content(record, requested_notes):
    note_type_to_column = {
        'Discharge Summary': 'Discharge_Summary_Text',
        'Progress Notes': 'Progress_Note_Text',
        'H&P': 'HP_Note_Text',
        'Echo': 'Echo_Text',
        'Imaging': 'Imaging_Text',
        'Consult': 'Consult_Text',
        'ECG': 'ECG_Text',
    }
    
    def truncate_text(text, max_chars=3000):
        if text and len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
    
    notes_content = ""
    for note_type in requested_notes:
        column_name = note_type_to_column.get(note_type)
        if column_name and column_name in df.columns and not pd.isna(df.loc[record, column_name]):
            notes_content += f"\n\n{note_type}:\n{truncate_text(df.loc[record, column_name])}"
    return notes_content

def predict_dispositions(records, llm_pipeline):
    predictions = []
    notes_requested_list = []

    for record in records:
        chief_complaint = df.loc[record, 'primarychiefcomplaintname']
        sex = df.loc[record, 'sex'] if 'sex' in df.columns else "Unknown"
        age = df.loc[record, 'Age'] if 'Age' in df.columns else "Unknown"
        presentation = df.loc[record, 'ED_Presentations'] if 'ED_Presentations' in df.columns else ""
        
        if pd.isna(chief_complaint):
            predictions.append(None)
            notes_requested_list.append("")
            continue

        requested_notes, selection_text = select_notes(record, llm_pipeline)
        notes_content = prepare_notes_content(record, requested_notes)

        message = [
            {
                "role": "system",
                "content": "You are an experienced ED physician. Your task is to predict the patient's ED disposition."
            },
            {
                "role": "user",
                "content": f"Chief Complaint: {chief_complaint}\nAge: {age}\nSex: {sex}\nPresentation: {presentation}\n{notes_content}\n"
                           f"Based on this information, choose the **most appropriate ED disposition** from the list below. "
                            "Only return the **exact text** of one disposition (no explanation):\n\n"
                            "1. Admit\n"
                            "2. Observation\n"
                            "3. OR Admit\n"
                            "4. Transfer to Another Facility\n"
                            "5. Eloped\n"
                            "6. AMA\n"
                            "7. LWBS after Triage\n"
                            "8. Send to L&D\n"
                            "9. Expired\n"
                            "10. Dismissed - Never Arrived\n"
                            "11. Discharge\n\n"
                            "12. None\n"
        
            }
        ]

        formatted_prompt = format_messages_for_llama(message)
        outputs = llm_pipeline(formatted_prompt, max_new_tokens=200, do_sample=False, temperature=0.5)
        generated_text = outputs[0]["generated_text"]
        predictions.append(generated_text)
        notes_requested_list.append(", ".join(requested_notes))
    
    return predictions, notes_requested_list

# Main function to process all rows
if __name__ == '__main__':

    valid_indices = df[df['primarychiefcomplaintname'].notna()].index  # Filter valid indices

    all_predictions = []
    all_requested_notes = []

    # Process all records without batching
    all_predictions, all_requested_notes = predict_dispositions(valid_indices, pipeline)

    # Assign predictions and requested notes to the corresponding rows in the DataFrame
    df.loc[valid_indices, 'Predicted_Disposition'] = all_predictions
    df.loc[valid_indices, 'Requested_Notes'] = all_requested_notes

    df.to_csv("...llama_disposition_predictions.csv", index=False)
    print("Batch predictions saved to 'llama_disposition_predictions.csv'")
