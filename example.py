"""
Example script demonstrating the Hybrid LLM Hallucination Detection System.
"""

import matplotlib.pyplot as plt

from analyzer import HallucinationAnalyzer


def run_example():
    """Run a simple example analysis."""
    print("Initializing Hybrid LLM Hallucination Detection System...")
    print("This may take a moment on first run (downloading models)...\n")

    analyzer = HallucinationAnalyzer(
        model_name="gpt2",
        semantic_threshold=0.80,
    )

    prompt = "What is the capital of France?"

    results = analyzer.analyze(
        prompt=prompt,
        num_responses=5,
        max_length=30,
        temperature=0.8,
    )

    analyzer.print_summary(results)

    print("\nDisplaying eigenvalue spectrum...")
    eigenvalues = results["eigen"].get("eigenvalues", [])
    if eigenvalues:
        analyzer.plot_eigenvalue_spectrum(
            eigenvalues,
            save_path="example_eigenvalue_spectrum.png",
        )
        plt.close("all")
        print("Eigenvalue spectrum saved to 'example_eigenvalue_spectrum.png'")
    else:
        print("No eigenvalues available to plot.")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nTo run the Streamlit UI, use: streamlit run app.py")
    print('To run custom prompts, use: python main.py --prompt "Your question here"')


if __name__ == "__main__":
    run_example()
