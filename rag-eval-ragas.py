import ollama
import pandas as pd
from typing import List, Dict, Tuple
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.testset import TestsetGenerator
from ragas.dataset import Dataset
from datasets import Dataset as HFDataset
import json

# Load the dataset (same as original RAG)
dataset = []
with open('cat-facts.txt', 'r') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

# RAG System Components (same as original)
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def get_rag_response(query: str) -> Tuple[str, List[str]]:
    """Get RAG response and return both answer and retrieved context as strings"""
    retrieved_knowledge = retrieve(query)
    
    # Extract just the text content from retrieved chunks
    context_chunks = [chunk.strip() for chunk, _ in retrieved_knowledge]
    context_text = '\n'.join([f' - {chunk}' for chunk in context_chunks])
    
    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}
'''
    
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query},
        ],
    )
    
    return response['message']['content'], context_chunks

# ===== RAGAS EVALUATION SYSTEM =====

# Test dataset for RAGAS evaluation
TEST_DATASET = [
    {
        "question": "Why do cats purr?",
        "ground_truth": "Cats purr for various reasons including contentment, healing, communication, and comfort. The purring sound is produced by rapid muscle contractions in the larynx and diaphragm."
    },
    {
        "question": "How many hours do cats sleep per day?",
        "ground_truth": "Cats typically sleep 12-16 hours per day, with some cats sleeping up to 20 hours. This is normal behavior for felines."
    },
    {
        "question": "What is a group of cats called?",
        "ground_truth": "A group of cats is called a clowder. Other terms include a glaring, pounce, or destruction of cats."
    },
    {
        "question": "Can cats see in the dark?",
        "ground_truth": "Cats have excellent night vision and can see in very low light conditions, though they cannot see in complete darkness. Their eyes are adapted for low-light vision."
    },
    {
        "question": "How long do cats typically live?",
        "ground_truth": "Indoor cats typically live 12-18 years, while outdoor cats have shorter lifespans of 2-5 years due to various risks and hazards."
    }
]

def create_ragas_dataset() -> HFDataset:
    """Create a dataset in the format required by RAGAS"""
    questions = []
    ground_truths = []
    answers = []
    contexts = []
    
    print("Creating RAGAS evaluation dataset...")
    
    for test_case in TEST_DATASET:
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        
        # Get RAG response
        answer, retrieved_contexts = get_rag_response(question)
        
        # Format for RAGAS
        questions.append(question)
        ground_truths.append([ground_truth])  # RAGAS expects list of ground truths
        answers.append(answer)
        contexts.append(retrieved_contexts)  # List of context strings
        
        print(f"Processed: {question}")
    
    # Create HuggingFace dataset
    ragas_dataset = HFDataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts
    })
    
    return ragas_dataset

def run_ragas_evaluation() -> Dict:
    """Run RAGAS evaluation on the dataset"""
    print("\n" + "="*60)
    print("RUNNING RAGAS EVALUATION")
    print("="*60)
    
    # Create dataset
    eval_dataset = create_ragas_dataset()
    
    print(f"\nDataset created with {len(eval_dataset)} samples")
    print("Sample from dataset:")
    print(f"Question: {eval_dataset[0]['question']}")
    print(f"Answer: {eval_dataset[0]['answer']}")
    print(f"Contexts: {eval_dataset[0]['contexts']}")
    
    # Define metrics to evaluate
    metrics = [
        faithfulness,           # How faithful is the answer to the context
        answer_relevancy,       # How relevant is the answer to the question
        context_precision,      # How precise is the retrieved context
        context_recall          # How much of the ground truth is covered by context
    ]
    
    print("\nRunning RAGAS evaluation with metrics:")
    print("- Faithfulness: How faithful is the answer to the context")
    print("- Answer Relevancy: How relevant is the answer to the question")
    print("- Context Precision: How precise is the retrieved context")
    print("- Context Recall: How much of the ground truth is covered by context")
    
    try:
        # Run evaluation
        result = evaluate(
            eval_dataset,
            metrics=metrics
        )
        
        return {
            "success": True,
            "results": result,
            "dataset_size": len(eval_dataset)
        }
        
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        return {
            "success": False,
            "error": str(e),
            "dataset_size": len(eval_dataset)
        }

def print_ragas_report(evaluation_result: Dict):
    """Print a formatted RAGAS evaluation report"""
    print("\n" + "="*60)
    print("RAGAS EVALUATION REPORT")
    print("="*60)
    
    if not evaluation_result["success"]:
        print(f"Evaluation failed: {evaluation_result['error']}")
        return
    
    results = evaluation_result["results"]
    dataset_size = evaluation_result["dataset_size"]
    
    print(f"Dataset Size: {dataset_size} samples")
    print(f"\nRAGAS Metrics (0-1 scale, higher is better):")
    
    # Extract and display metrics
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
        print("\nDetailed Results:")
        print(df.to_string(index=False))
        
        # Calculate averages
        avg_metrics = df.mean()
        print(f"\nAverage Scores:")
        for metric, score in avg_metrics.items():
            if metric != 'question':
                print(f"  {metric}: {score:.3f}")
    else:
        print("Results format not recognized. Raw results:")
        print(results)

def run_comparison_evaluation():
    """Run both custom LLM judge and RAGAS evaluation for comparison"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RAG EVALUATION COMPARISON")
    print("="*80)
    
    # Run RAGAS evaluation
    print("\n1. Running RAGAS Evaluation...")
    ragas_result = run_ragas_evaluation()
    print_ragas_report(ragas_result)
    
    # Run custom LLM judge evaluation (import from previous file)
    print("\n2. Running Custom LLM Judge Evaluation...")
    try:
        from rag_eval import run_evaluation, print_evaluation_report
        custom_result = run_evaluation()
        print_evaluation_report(custom_result)
    except ImportError:
        print("Custom evaluation not available. Run rag-eval.py separately.")
    
    return ragas_result

def interactive_ragas_mode():
    """Run RAG system in interactive mode with RAGAS evaluation option"""
    print("\n" + "="*50)
    print("INTERACTIVE RAG WITH RAGAS EVALUATION")
    print("="*50)
    print("Ask questions about cats! Type 'ragas' to run RAGAS evaluation, 'quit' to exit.")
    
    while True:
        query = input('\nAsk me a question: ').strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'ragas':
            results = run_ragas_evaluation()
            print_ragas_report(results)
            continue
        
        # Get RAG response
        answer, retrieved_contexts = get_rag_response(query)
        
        print('\nRetrieved knowledge:')
        for i, context in enumerate(retrieved_contexts):
            print(f' - Context {i+1}: {context}')
        
        print(f'\nChatbot response: {answer}')

if __name__ == "__main__":
    # Run RAGAS evaluation by default
    results = run_ragas_evaluation()
    print_ragas_report(results)
    
    # Then start interactive mode
    interactive_ragas_mode()