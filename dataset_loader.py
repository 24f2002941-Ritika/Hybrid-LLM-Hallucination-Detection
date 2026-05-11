"""
Dataset Loader Module
Loads and caches the TruthfulQA dataset as well as CoQA, SQuAD, NQ, and TriviaQA 
from HuggingFace for ground truth comparison.

Memory-optimised: caps each dataset at MAX_PER_SOURCE entries to stay within
the 16 GB RAM limit on Hugging Face Spaces free tier.
"""

from datasets import load_dataset
from typing import List, Dict

# Maximum QA pairs to retain per dataset source. This keeps total memory
# for embeddings + dataset objects well under 4 GB.
MAX_PER_SOURCE = 2000

def load_truthfulqa(split: str = "validation"):
    """
    Load the TruthfulQA dataset (generation config) from HuggingFace.
    """
    dataset = load_dataset("truthful_qa", "generation")
    return dataset[split]

def _cap(pairs: List[Dict], limit: int) -> List[Dict]:
    """Return at most *limit* items from *pairs*."""
    return pairs[:limit]

def get_all_qa_pairs(split: str = "validation") -> List[Dict]:
    """
    Load TruthfulQA, SQuAD, NQ Open, Trivia QA, and CoQA.
    Returns a unified list of dictionaries with 'question', 'best_answer', and 'source'.
    Each source is capped at MAX_PER_SOURCE entries to limit memory usage.
    """
    pairs = []
    
    # 1. TruthfulQA (small dataset, ~817 entries — no cap needed)
    try:
        tqa = load_truthfulqa(split)
        src = []
        for row in tqa:
            src.append({"question": row["question"], "best_answer": row["best_answer"], "source": "TruthfulQA"})
        pairs.extend(src)
        print(f"  TruthfulQA: {len(src)} entries loaded")
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")

    # 2. SQuAD (large — cap it)
    try:
        print("Loading SQuAD dataset...")
        squad = load_dataset("squad", split=split)
        src = []
        for row in squad:
            ans = row["answers"]["text"][0] if row["answers"]["text"] else ""
            if ans:
                src.append({"question": row["question"], "best_answer": ans, "source": "SQuAD"})
            if len(src) >= MAX_PER_SOURCE:
                break
        pairs.extend(src)
        print(f"  SQuAD: {len(src)} entries loaded (capped at {MAX_PER_SOURCE})")
    except Exception as e:
        print(f"Error loading SQuAD: {e}")

    # 3. NQ Open
    try:
        print("Loading NQ Open dataset...")
        nq = load_dataset("nq_open", split=split)
        src = []
        for row in nq:
            ans = row["answer"][0] if row["answer"] else ""
            if ans:
                src.append({"question": row["question"], "best_answer": ans, "source": "NQ"})
            if len(src) >= MAX_PER_SOURCE:
                break
        pairs.extend(src)
        print(f"  NQ: {len(src)} entries loaded (capped at {MAX_PER_SOURCE})")
    except Exception as e:
        print(f"Error loading NQ: {e}")

    # 4. Trivia QA
    try:
        print("Loading Trivia QA dataset...")
        trivia = load_dataset("trivia_qa", "rc.nocontext", split=split)
        src = []
        for row in trivia:
            ans = row["answer"]["value"]
            if ans:
                src.append({"question": row["question"], "best_answer": ans, "source": "TriviaQA"})
            if len(src) >= MAX_PER_SOURCE:
                break
        pairs.extend(src)
        print(f"  TriviaQA: {len(src)} entries loaded (capped at {MAX_PER_SOURCE})")
    except Exception as e:
        print(f"Error loading TriviaQA: {e}")

    # 5. CoQA
    try:
        print("Loading CoQA dataset...")
        coqa = load_dataset("coqa", split=split)
        src = []
        for row in coqa:
            questions = row["questions"]
            answers = row["answers"]["input_text"]
            for q, a in zip(questions, answers):
                src.append({"question": q, "best_answer": a, "source": "CoQA"})
                if len(src) >= MAX_PER_SOURCE:
                    break
            if len(src) >= MAX_PER_SOURCE:
                break
        pairs.extend(src)
        print(f"  CoQA: {len(src)} entries loaded (capped at {MAX_PER_SOURCE})")
    except Exception as e:
        print(f"Error loading CoQA: {e}")

    print(f"Total QA pairs loaded: {len(pairs)}")
    return pairs

def build_qa_lookup(split: str = "validation") -> dict:
    """
    Return a dict mapping each question (lowercased) -> best_answer.
    Useful for fast exact-lookup.
    """
    data = load_truthfulqa(split)
    return {entry["question"].strip().lower(): entry["best_answer"] for entry in data}