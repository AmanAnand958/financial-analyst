import os
import json
import re
import time
import pdfplumber
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Setup Groq Client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=api_key)
MODEL_NAME = "llama-3.1-8b-instant"

PDF_PATH = "data/raw/kpmg_mock_annual_report_fy2023.pdf"
OUTPUT_PATH = "data/processed/ft_dataset.json"

PROMPT_TEMPLATE = """You are a machine learning data pipeline generator. Your task is to generate exactly 5 high-quality instruction-tuning QA pairs based ONLY on the provided financial context page. 

The training pairs must be formatted as a JSON list of dictionaries. Each dictionary must contain:
1. "instruction": A natural question, reconciliation request, or auditing task that a senior auditor or financial analyst would ask.
2. "input": The exact context snippet from the page that contains the answers, representing the retrieved RAG context.
3. "output": A factual, detailed, professional answer that strictly references the figures, notes, or pages in the context, adopting a formal auditing tone.

Here is the context page:
---
{page_text}
---

Generate exactly 5 QA pairs. Diversify the instructions (factual lookups, calculations, footnote lookups, or out-of-context edge cases).
Return ONLY the raw JSON array. Do not include markdown code block formatting (like ```json), intro, or outro text.
"""

def extract_pdf_pages(pdf_path):
    print(f"Extracting text from {pdf_path}...")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((idx + 1, text))
    return pages

def generate_dataset():
    if not os.path.exists(PDF_PATH):
        print(f"Error: Mock PDF not found at {PDF_PATH}. Please run generate_pdf.py first.")
        return

    pages = extract_pdf_pages(PDF_PATH)
    all_dataset = []

    for page_num, text in pages:
        print(f"\n--- Processing Page {page_num}/{len(pages)} ---")
        
        # We run 3 mini-batches of 5 QA pairs per page to get 15 high-quality pairs per page
        for batch_idx in range(3):
            print(f"  Generating Batch {batch_idx + 1}/3...")
            prompt = PROMPT_TEMPLATE.format(page_text=text)
            
            retries = 3
            while retries > 0:
                try:
                    # Add a sleep to prevent rate limiting (30 RPM limit)
                    time.sleep(2.5)
                    
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a database generator that outputs raw JSON data arrays only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                    )
                    
                    response_content = completion.choices[0].message.content.strip()
                    # Clean up any potential markdown wraps
                    if response_content.startswith("```"):
                        response_content = re.sub(r'^```json\s*|\s*```$', '', response_content, flags=re.MULTILINE)
                    
                    # Basic JSON repair: clean up trailing commas before closing brackets
                    response_content = re.sub(r',\s*\]', ']', response_content)
                    response_content = re.sub(r',\s*\}', '}', response_content)
                    
                    parsed_json = json.loads(response_content)
                    if isinstance(parsed_json, list):
                        print(f"    Generated {len(parsed_json)} QA pairs.")
                        all_dataset.extend(parsed_json)
                        break
                    else:
                        print("    Error: Output is not a list. Retrying...")
                        retries -= 1
                except Exception as e:
                    print(f"    Error in batch {batch_idx + 1}: {e}. Retrying in 5s...")
                    time.sleep(5.0)
                    retries -= 1
            
            if retries == 0:
                print(f"    Failed batch {batch_idx + 1}.")

    if all_dataset:
        print(f"\nWriting {len(all_dataset)} total QA pairs to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_dataset, f, indent=2)
        print("Dataset generation complete!")
    else:
        print("No QA pairs were generated.")

if __name__ == "__main__":
    generate_dataset()
