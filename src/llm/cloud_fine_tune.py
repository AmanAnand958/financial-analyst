import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Setup OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # If the user doesn't have an OpenAI key set, we will guide them to set it.
    print("Warning: OPENAI_API_KEY environment variable is not set. Please set it in your .env file to run this script.")

client = OpenAI(api_key=api_key) if api_key else None

INPUT_PATH = "data/processed/ft_dataset.json"
OUTPUT_JSONL_PATH = "data/processed/ft_dataset_openai.jsonl"
SYSTEM_PROMPT = "You are a financial analyst assistant helping KPMG auditors. Respond strictly based on the provided context."

def convert_to_openai_format():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Base dataset not found at {INPUT_PATH}. Please generate the dataset first.")
        return False

    print(f"Loading base dataset from {INPUT_PATH}...")
    with open(INPUT_PATH, "r") as f:
        dataset = json.load(f)

    print(f"Converting {len(dataset)} pairs to OpenAI JSONL format...")
    with open(OUTPUT_JSONL_PATH, "w") as f:
        for item in dataset:
            # Format user content with context input and question instruction
            user_content = f"Context: {item['input']}\n\nQuestion: {item['instruction']}"
            
            messages_structure = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item["output"]}
                ]
            }
            f.write(json.dumps(messages_structure) + "\n")
            
    print(f"Successfully wrote OpenAI-formatted dataset to {OUTPUT_JSONL_PATH}.")
    return True

def trigger_cloud_fine_tuning():
    if not client:
        print("Aborting cloud trigger: OpenAI Client not initialized (missing API key).")
        return

    if not os.path.exists(OUTPUT_JSONL_PATH):
        print(f"Error: Converted dataset not found at {OUTPUT_JSONL_PATH}. Run conversion first.")
        return

    print("Uploading dataset to OpenAI...")
    try:
        with open(OUTPUT_JSONL_PATH, "rb") as file_data:
            uploaded_file = client.files.create(
                file=file_data,
                purpose="fine-tune"
            )
        file_id = uploaded_file.id
        print(f"File uploaded successfully. File ID: {file_id}")
        
        print("Launching fine-tuning job on gpt-4o-mini-2024-07-18...")
        ft_job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-4o-mini-2024-07-18"
        )
        print(f"Fine-tuning job triggered successfully!")
        print(f"Job ID: {ft_job.id}")
        print(f"Status: {ft_job.status}")
        print("\nYou can monitor the job progress in your OpenAI developer dashboard (https://platform.openai.com/finetuning).")
        print("Once the job is completed, you will receive an email and can query your custom model.")
    except Exception as e:
        print(f"Failed to trigger fine-tuning: {e}")

def run_fine_tuned_inference(model_id, prompt, context):
    if not client:
        print("API key missing. Cannot query OpenAI.")
        return
        
    user_content = f"Context: {context}\n\nQuestion: {prompt}"
    print(f"Querying fine-tuned model '{model_id}'...")
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1
        )
        print("\n--- Response ---")
        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"Inference error: {e}")

if __name__ == "__main__":
    if convert_to_openai_format():
        # Triggering is optional and requires a valid API key
        if os.getenv("OPENAI_API_KEY"):
            trigger_cloud_fine_tuning()
        else:
            print("\nTo trigger the fine-tuning job in the cloud, set your 'OPENAI_API_KEY' in the .env file.")
