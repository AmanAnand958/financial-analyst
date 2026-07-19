import os
import time
import json
from dotenv import load_dotenv
load_dotenv()

from src.ingestion.document_processor import DocumentProcessor
from src.retrieval.hybrid_search import HybridRetriever
from src.llm.generator import FinancialAnswerGenerator

# Setup real testing document details
REAL_PDF = "data/raw/microsoft_10k_2023.pdf"
DOCUMENT_ID = "microsoft_10k_2023"

TEST_QUERIES = [
    {
        "question": "What was Microsoft's total revenue in fiscal year 2023?",
        "expected": "211,915 million"
    },
    {
        "question": "What was the operating income of Microsoft in FY2023?",
        "expected": "88,523 million"
    },
    {
        "question": "What was the total revenue for the Intelligent Cloud segment in FY2023?",
        "expected": "87,907 million"
    },
    {
        "question": "What was Microsoft's net income for fiscal year 2023?",
        "expected": "72,361 million"
    }
]

def run_real_system_test():
    print("======================================================================")
    print("         FinDoc Intelligence — Real Document Evaluation Test          ")
    print("======================================================================")
    print(f"Target PDF: {REAL_PDF}")
    
    if not os.path.exists(REAL_PDF):
        print(f"Error: PDF not found at {REAL_PDF}. Download failed or path is invalid.")
        return

    print("\n[Step 1] Ingesting real Microsoft 10-K report (85 pages)...")
    print("This will extract, chunk, and index the entire document in ChromaDB.")
    
    processor = DocumentProcessor()
    retriever = HybridRetriever()

    # Reset collection state for Microsoft document
    try:
        retriever.collection.delete(where={"document_id": DOCUMENT_ID})
        print("Cleared previous indices for a clean state.")
    except Exception as e:
        print(f"Note clearing previous index: {e}")

    start_time = time.time()
    result = processor.process_document(REAL_PDF, f"{DOCUMENT_ID}.pdf")
    if not result["success"]:
        print(f"Error: Processing failed: {result.get('error')}")
        return

    print(f"Extracted {len(result['chunks'])} chunks from PDF in {time.time() - start_time:.2f} seconds.")

    print("\n[Step 2] Storing chunks in ChromaDB vector store...")
    retriever.add_chunks(result["chunks"])
    print("Vector database indexing complete.")

    # Re-initialize retriever to load the new BM25 index
    retriever = HybridRetriever()
    generator = FinancialAnswerGenerator(confidence_threshold=0.45)

    print("\n[Step 3] Running RAG System Test Queries...")
    print("Retrieving chunks -> Reranking -> Querying LLM -> Applying NCV Guard...")
    
    for idx, test_case in enumerate(TEST_QUERIES):
        print(f"\n---------------------------------------------------------")
        print(f"Test Query #{idx + 1}: {test_case['question']}")
        print(f"Expected Answer contains: {test_case['expected']}")
        print(f"---------------------------------------------------------")
        
        # Retrieve candidate chunks
        retrieved_chunks = retriever.hybrid_search(test_case["question"], top_k=5)
        print(f"Retrieved {len(retrieved_chunks)} candidate chunks.")
        
        # Generate Answer
        response = generator.generate(
            question=test_case["question"],
            context_chunks=retrieved_chunks,
            use_ncv=True
        )
        
        print("\n--- System Response ---")
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']}")
        print(f"NCV Status: {'PASSED' if response['confidence'] == 'HIGH' else 'FAILED/ABSTAINED'}")
        print(f"Latency: {response['latency_ms']}ms")

if __name__ == "__main__":
    run_real_system_test()
