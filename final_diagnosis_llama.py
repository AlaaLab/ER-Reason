import sys
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, pipeline
import math
import time
import transformers
import torch
import pandas as pd
import json
import gc
import os

df = pd.read_csv('/filtered_final_dataset.csv')
output_csv= "/diag_predictions.csv"

model_path = '/wynton/protected/project/shared_models/llama3-hf_series/Llama-3.2-3B-Instruct/'
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    device_map="auto"
)

# === Prompt Formatting ===
def format_messages_for_llama(messages):
    prompt = ""
    for message in messages:
        prompt += f"<|{message['role']}|>\n{message['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt

def truncate_text(text, max_chars=3000):
    return text[:max_chars] + "..." if text and len(text) > max_chars else text

def truncate_prompt_to_fit(prompt, max_tokens=3840):
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# === Note Selection ===
def select_notes(record, df, llm_pipeline):
    try:
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
            {"role": "system", "content": "You are an experienced Emergency Department (ED) physician. Your task is to decide which medical notes you need to read to predict the patient's ED primary diagnosis."},
            {"role": "user", "content": f"Patient: {age}yo {sex}, Chief Complaint: {chief_complaint}\n\n" +
                "\n".join([f"- {note}: {'Available' if available else 'Not available'}" for note, available in available_notes.items()]) +
                "\nList ONLY the note types you need to review (comma-separated, no explanation). Always include Discharge Summary if available."}
        ]

        prompt = format_messages_for_llama(selection_messages)
        prompt = truncate_prompt_to_fit(prompt)

        outputs = llm_pipeline(prompt, max_new_tokens=100, do_sample=False, temperature=0.5)
        selection_text = outputs[0]["generated_text"]

        if "<|assistant|>" in selection_text:
            selection_text = selection_text.split("<|assistant|>")[-1].strip()
        selection_text = selection_text.replace("</s>", "").strip()

        requested_notes = [n.strip() for n in selection_text.split(',') if n.strip() in available_notes and available_notes[n.strip()]]

        if available_notes['Discharge Summary'] and 'Discharge Summary' not in requested_notes:
            requested_notes.append('Discharge Summary')
        if not requested_notes:
            requested_notes = [k for k, v in available_notes.items() if v][:2]

        return requested_notes, selection_text
    except Exception as e:
        print(f"[Error in select_notes] Record {record}: {e}")
        return [], "Selection Failed"

# === Note Prep ===
def prepare_notes_content(record, requested_notes, df):
    note_type_to_column = {
        'Discharge Summary': 'Discharge_Summary_Text',
        'Progress Notes': 'Progress_Note_Text',
        'H&P': 'HP_Note_Text',
        'Echo': 'Echo_Text',
        'Imaging': 'Imaging_Text',
        'Consult': 'Consult_Text',
        'ECG': 'ECG_Text',
    }
    notes_content = ""
    for note_type in requested_notes:
        col = note_type_to_column.get(note_type)
        if col and col in df.columns and not pd.isna(df.loc[record, col]):
            notes_content += f"\n\n{note_type}:\n{truncate_text(df.loc[record, col])}"
    return notes_content

# === Main Function (No Batching) ===
def run_predictions(df, output_csv):
    for col in ['Predicted_Diagnosis', 'Requested_Notes', 'Prediction_Correct']:
        if col not in df.columns:
            df[col] = None

    if os.path.exists(output_csv):
        print(f"🔁 Resuming from {output_csv}")
        df_saved = pd.read_csv(output_csv)
        df.update(df_saved)

    for idx in df.index:
        if pd.notna(df.loc[idx, 'Predicted_Diagnosis']):
            continue

        try:
            requested_notes, selection_text = select_notes(idx, df, pipeline)
            notes_content = prepare_notes_content(idx, requested_notes, df)

            chief_complaint = df.loc[idx, 'primarychiefcomplaintname']
            sex = df.loc[idx, 'sex'] if 'sex' in df.columns else "Unknown"
            age = df.loc[idx, 'Age'] if 'Age' in df.columns else "Unknown"
            presentation = df.loc[idx, 'presentation'] if 'presentation' in df.columns else "Unknown"

            messages = [
                {"role": "system", "content": "You are an experienced ED physician predicting a patient's diagnosis."},
                {"role": "user", "content": f"Chief Complaint: {chief_complaint}\nAge: {age}\nSex: {sex}\nPresentation: {presentation}\n{notes_content}\n\nPredict the single most likely ED diagnosis (in words only, no code)."}
            ]

            prompt = format_messages_for_llama(messages)
            prompt = truncate_prompt_to_fit(prompt)

            output = pipeline(prompt, max_new_tokens=256, do_sample=False, temperature=0.5)[0]["generated_text"]
            if "<|assistant|>" in output:
                output = output.split("<|assistant|>")[-1].strip()
            output = output.replace("</s>", "").strip()

            df.loc[idx, 'Predicted_Diagnosis'] = output
            df.loc[idx, 'Requested_Notes'] = ", ".join(requested_notes)
            df.loc[idx, 'Prediction_Correct'] = output == df.loc[idx, 'primaryeddiagnosisname']

        except Exception as e:
            print(f"[Error] Record {idx}: {e}")
            df.loc[idx, 'Predicted_Diagnosis'] = "ERROR"
            df.loc[idx, 'Requested_Notes'] = "ERROR"
            df.loc[idx, 'Prediction_Correct'] = False

        df.to_csv(output_csv, index=False)

    acc = df["Prediction_Correct"].mean() * 100
    print(f"\n✅ Done. Accuracy: {acc:.2f}%")
    print(f"📄 Results saved to: {output_csv}")



run_predictions(df, "/diag_predictions.csv")

