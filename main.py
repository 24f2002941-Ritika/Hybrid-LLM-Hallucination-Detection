"""
Main Module.
Command-line interface for the Hybrid LLM Hallucination Detection System.
"""

import argparse

from datasets import load_dataset

from analyzer import HallucinationAnalyzer


def main():
    """Main entry point for the hallucination detection system."""
    parser = argparse.ArgumentParser(
        description="Hybrid LLM Hallucination Detection System"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional single prompt to analyze (if not using full dataset)",
    )
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Run evaluation on the TruthfulQA validation set",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=5,
        help="Number of responses to generate (default: 5)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum generation length (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="GPT-2 model variant (default: gpt2)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Weight for internal risk in final score (default: 0.6)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="Weight for external risk in final score (default: 0.4)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.80,
        help="Min cosine similarity to match a TruthfulQA question (default: 0.80)",
    )

    args = parser.parse_args()

    analyzer = HallucinationAnalyzer(
        model_name=args.model,
        semantic_threshold=args.semantic_threshold,
    )

    if args.use_dataset:
        dataset = load_dataset("truthful_qa", "generation")
        data = dataset["validation"]

        for example in data:
            question = example["question"]

            print("\n" + "=" * 30)
            print(f"Question: {question}")
            print("=" * 30)

            results = analyzer.analyze(
                prompt=question,
                num_responses=args.num_responses,
                max_length=args.max_length,
                temperature=args.temperature,
                alpha=args.alpha,
                beta=args.beta,
            )
            analyzer.print_summary(results)
    elif args.prompt:
        results = analyzer.analyze(
            prompt=args.prompt,
            num_responses=args.num_responses,
            max_length=args.max_length,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
        )
        analyzer.print_summary(results)
    else:
        print("Provide --prompt or --use-dataset")


if __name__ == "__main__":
    main()
