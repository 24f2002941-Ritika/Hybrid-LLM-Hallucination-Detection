"""
Dataset Loader Module
Loads and caches the TruthfulQA dataset as well as CoQA, SQuAD, NQ, and TriviaQA 
from HuggingFace for ground truth comparison.
"""

from datasets import load_dataset
from typing import List, Dict

def load_truthfulqa(split: str = "validation"):
    """
    Load the TruthfulQA dataset (generation config) from HuggingFace.
    """
    dataset = load_dataset("truthful_qa", "generation")
    return dataset[split]

def get_all_qa_pairs(split: str = "validation") -> List[Dict]:
    """
    Load TruthfulQA, SQuAD, NQ Open, Trivia QA, and CoQA.
    Returns a unified list of dictionaries with 'question', 'best_answer', and 'source'.
    """
    pairs = []
    
    # 1. TruthfulQA
    try:
        tqa = load_truthfulqa(split)
        for row in tqa:
            pairs.append({"question": row["question"], "best_answer": row["best_answer"], "source": "TruthfulQA"})
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")

    # 2. SQuAD
    try:
        print("Loading SQuAD dataset...")
        squad = load_dataset("squad", split=split)
        for row in squad:
            ans = row["answers"]["text"][0] if row["answers"]["text"] else ""
            if ans:
                pairs.append({"question": row["question"], "best_answer": ans, "source": "SQuAD"})
    except Exception as e:
        print(f"Error loading SQuAD: {e}")

    # 3. NQ Open
    try:
        print("Loading NQ Open dataset...")
        nq = load_dataset("nq_open", split=split)
        for row in nq:
            ans = row["answer"][0] if row["answer"] else ""
            if ans:
                pairs.append({"question": row["question"], "best_answer": ans, "source": "NQ"})
    except Exception as e:
        print(f"Error loading NQ: {e}")

    # 4. Trivia QA
    try:
        print("Loading Trivia QA dataset...")
        trivia = load_dataset("trivia_qa", "rc.nocontext", split=split)
        for row in trivia:
            ans = row["answer"]["value"]
            if ans:
                pairs.append({"question": row["question"], "best_answer": ans, "source": "TriviaQA"})
    except Exception as e:
        print(f"Error loading TriviaQA: {e}")

    # 5. CoQA
    try:
        print("Loading CoQA dataset...")
        coqa = load_dataset("coqa", split=split)
        for row in coqa:
            questions = row["questions"]
            answers = row["answers"]["input_text"]
            for q, a in zip(questions, answers):
                pairs.append({"question": q, "best_answer": a, "source": "CoQA"})
    except Exception as e:
        print(f"Error loading CoQA: {e}")

    return pairs

def build_qa_lookup(split: str = "validation") -> dict:
    """
    Return a dict mapping each question (lowercased) -> best_answer.
    Useful for fast exact-lookup.
    """
    data = load_truthfulqa(split)
    return {entry["question"].strip().lower(): entry["best_answer"] for entry in data}