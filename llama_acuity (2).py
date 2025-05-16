import sys
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import json
import gc

# Load the CSV file
df = pd.read_csv('acuity_results.csv')

# Path to the model
model_path = '/wynton/protected/project/shared_models/llama3-hf_series/Llama-3.2-3B-Instruct/'

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
def process_record(idx, row, model, tokenizer, temperature=TEMPERATURE):
    # Prepare the input message
    messages = [
        {
            "role": "system",
            "content": "You are an experienced Triage Nurse in the ER. Predict the emergency department acuity level for this patient."
        },
        {
            "role": "user",
            "content": f"Select the most appropriate acuity level from the following options ONLY:\n'Immediate', 'Emergent', 'Urgent', 'Less Urgent', 'Non-Urgent'\n\nRespond with ONLY ONE of these five options based on the information provided.\nChief Complaint: {row['primarychiefcomplaintname']}\nVitals: {row['Vital_Signs']}\nAge: {row['Age']}\nSex: {row['sex']}\nRace: {row['firstrace']}"
        },
    ]
    
    # Format messages for Llama 3
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate output using the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temperature,
        do_sample=(temperature > 0)  # Enable sampling only if temperature > 0
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
if __name__ == '__main__':  # Fixed the syntax here
    # Process each record in the DataFrame
    summaries = []
    
    # Add a progress indicator
    total_records = len(df)
    print(f"Processing {total_records} records with temperature={TEMPERATURE}...")
    
#     # Process a small sample first to verify output format
#     sample_size = min(3, len(df))
#     print(f"Testing with {sample_size} sample records first...")
    
#     for idx in range(sample_size):
#         row = df.iloc[idx]
#         summary = process_record(idx, row, model, tokenizer)
#         print(f"\nSample {idx+1}:")
#         print(f"Response: {summary}")
    
#     # Ask for confirmation before processing the full dataset
#     proceed = input("\nContinue with full dataset? (y/n): ")
#     if proceed.lower() != 'y':
#         print("Exiting...")
#         sys.exit()
    
    # Process all records
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing record {idx}/{total_records}")
        
        summary = process_record(idx, row, model, tokenizer)
        summaries.append(summary)
        
        # Occasional garbage collection to manage memory
        if idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Add the generated summaries as a new column in the DataFrame
    df['Predicted_Acuity'] = summaries
    
    # Save the updated DataFrame to a new CSV file - include temperature in filename
    output_filename = f'RACE_llama_4_experiment_temp{TEMPERATURE}.csv'
    df.to_csv(output_filename, index=False)
    print(f"Processed summaries saved to '{output_filename}'")