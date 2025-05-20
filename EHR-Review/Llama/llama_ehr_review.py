################################################################################################
# Job for Llama 3 Baseline
# Author: Niloufar Golchini
# Date last edited: 03/10/2025
################################################################################################


import sys
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
import gc

# Load the CSV file
discharge_summaries = pd.read_csv('results.csv')

# Path to the model
model_path = 'Llama-3.2-3B-Instruct/'

# Model parameters
TEMPERATURE = 0.5  
MAX_NEW_TOKENS = 256

# Initialize the tokenizer and model directly instead of using pipeline
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16  # Use bfloat16 for efficiency
)

# Function to process each record
def process_record(record_idx, model, tokenizer):
    # Prepare the input message
    messages = [
        {
            "role": "system",
            "content": "You are an experienced emergency department (ED) physician creating a one-liner for a NEW patient who has just arrived at the ED. The patient's past discharge summary is available to you.\n"
        },
        {
            "role": "user",
            "content": f"Your task is to summarize the patient's relevant PAST medical history and end with their CURRENT chief complaint that is given with no adjectives about the chief complaint as you can NOT assume anything about their current condition. All notes and medical records provided are from PAST encounters, not the current visit.Create a concise one-liner summary for a patient who has just arrived at the Emergency Department. The one-liner must: 1. Start with demographic information (age, sex). Example of a one liner:  80 y.o. old male, with h/o of HFpEF (EF 55-60% 05/20/22), HTN, HLD, and bipolar disorder presenting with shortness of breath.  Include a concise summary of relevant PAST medical history from previous visits/notes and end with the current Chief Complaint: {discharge_summaries.loc[record_idx, 'primarychiefcomplaintname']}\nDischarge Summary: {discharge_summaries.loc[record_idx, 'Discharge_Summary_Text']}\n Age: {discharge_summaries.loc[record_idx,'Age']}\n Sex: {discharge_summaries.loc[record_idx,'sex']}"
        },
    ]

    # Format messages for Llama 3
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate output using the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=(TEMPERATURE > 0)  # Enable sampling only if temperature > 0
    )
    
    # Extract the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after the prompt)
    if "[/INST]" in generated_text:
        response = generated_text.split("[/INST]")[-1].strip()
    else:
        # Fallback extraction method
        prompt_end = prompt[-30:]  # Use last part of prompt as marker
        if prompt_end in generated_text:
            response = generated_text.split(prompt_end, 1)[-1].strip()
        else:
            response = generated_text  # Fallback to full text
    
    return response

# Main function
if __name__ == '__main__':
    # Process each record in the DataFrame
    summaries = []
    
    # Add a progress indicator
    total_records = len(discharge_summaries)
    print(f"Processing {total_records} records with temperature={TEMPERATURE}...")
    
    # Process all records
    for idx in range(len(discharge_summaries)):
        if idx % 10 == 0:
            print(f"Processing record {idx}/{total_records}")
        
        summary = process_record(idx, model, tokenizer)
        summaries.append(summary)
        
        # Occasional garbage collection to manage memory
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Add the generated summaries as a new column in the DataFrame
    discharge_summaries['Generated_Summary'] = summaries
    
    # Save the updated DataFrame to a new CSV file - include temperature in filename
    output_filename = f'llama_ehr_temp{TEMPERATURE}.csv'
    discharge_summaries.to_csv(output_filename, index=False)
    print(f"Processed summaries saved to '{output_filename}'")
