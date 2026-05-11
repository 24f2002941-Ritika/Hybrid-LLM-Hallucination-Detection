"""
Analyzer Module.
Combines internal and external metrics to compute the final hallucination risk score.
Supports GPT-2 and GPT-Neo variants via TransformerLens.
"""

from typing import Dict, Any, Optional
from model_loader import GPT2ModelLoader
from internal_metrics import InternalMetrics
from external_verifier import ExternalVerifier
import matplotlib.pyplot as plt
import numpy as np

# All supported TransformerLens models used by the app.
SUPPORTED_MODELS = {
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/pythia-2.8b",
    "facebook/opt-6.7b",
}


class HallucinationAnalyzer:
    """
    Main analyzer that combines all metrics for hallucination detection.
    Supports GPT-2, GPT-Neo, Pythia, and OPT variants via TransformerLens.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        semantic_threshold: float = 0.80
    ):
        """
        Initialize the analyzer with all components.

        Args:
            model_name: Model name for a supported TransformerLens model
            semantic_threshold: Minimum cosine similarity for TruthfulQA question
                                matching (0-1). Lower values allow fuzzier matches.
        """
        self.model_name = model_name

        if model_name not in SUPPORTED_MODELS:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

        # TransformerLens model path
        self.model_loader = GPT2ModelLoader(model_name)
        self.internal_metrics = InternalMetrics(self.model_loader.get_model())

        # ExternalVerifier loads TruthfulQA directly from HuggingFace
        self.external_verifier = ExternalVerifier(
            semantic_threshold=semantic_threshold
        )
    
    def analyze(
        self,
        prompt: str,
        num_responses: int = 5,
        max_length: int = 50,
        temperature: float = 0.8,
        alpha: float = 0.6,
        beta: float = 0.4,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Complete hallucination analysis pipeline.

        Args:
            prompt: Input prompt/question
            num_responses: Number of responses to generate
            max_length: Maximum generation length
            temperature: Sampling temperature
            alpha: Weight for internal risk in final score
            beta: Weight for external risk in final score
            w1, w2, w3: Weights for eigen score, stability, grounding

        Returns:
            Dictionary with all metrics and results
        """
        print("\n" + "=" * 80)
        print("HYBRID LLM HALLUCINATION DETECTION SYSTEM")
        print("=" * 80)
        print(f"\nPrompt: {prompt}\n")

        return self._analyze_gpt2(
            prompt, num_responses, max_length, temperature,
            alpha, beta, w1, w2, w3
        )

    # ------------------------------------------------------------------
    # GPT-2 analysis path (original, unchanged logic)
    # ------------------------------------------------------------------

    def _analyze_gpt2(
        self,
        prompt: str,
        num_responses: int,
        max_length: int,
        temperature: float,
        alpha: float,
        beta: float,
        w1: float,
        w2: float,
        w3: float,
    ) -> Dict[str, Any]:
        # Step 1: Generate multiple responses
        print("Step 1: Generating responses...")
        responses = self.model_loader.generate_responses(
            prompt=prompt,
            num_responses=num_responses,
            max_length=max_length,
            temperature=temperature,
        )

        # Step 2: Generate primary response with cache for internal analysis
        print("\nStep 2: Generating primary response with activations...")
        primary_generation = self.model_loader.generate_with_cache(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
        )

        # Step 3: Compute internal metrics
        print("\nStep 3: Computing internal metrics...")

        # EigenScore: pass the K sampled responses (INSIDE-paper implementation)
        eigen_metrics = self.internal_metrics.compute_eigen_score(responses)
        print(f"  Eigen Score:     {eigen_metrics['eigen_score']:.4f}")
        print(f"  Responses used:  {eigen_metrics['num_responses']}")

        stability_metrics = self.internal_metrics.compute_stability(
            primary_generation["cache"],
            primary_generation["prompt_length"],
        )
        print(f"  Stability Score: {stability_metrics['stability_score']:.4f}")

        total_length = primary_generation["tokens"].shape[0]
        grounding_metrics = self.internal_metrics.compute_attention_grounding(
            primary_generation["cache"],
            primary_generation["prompt_length"],
            total_length,
        )
        print(f"  Grounding Score: {grounding_metrics['grounding_score']:.4f}")

        internal_risk_metrics = self.internal_metrics.compute_internal_risk(
            eigen_metrics, stability_metrics, grounding_metrics, w1, w2, w3
        )
        print(f"  Internal Risk: {internal_risk_metrics['internal_risk']:.4f}")

        # Step 4: Compute external metrics
        print("\nStep 4: Computing external metrics...")
        external_metrics = self.external_verifier.compute_external_metrics(
            prompt, responses
        )

        if external_metrics is None:
            print("  Warning: No ground truth available, using default external risk")
            external_metrics = {
                "similarities": [0.5] * num_responses,
                "external_consistency": 0.5,
                "external_risk": 0.5,
                "ground_truth": "N/A",
                "ground_truth_source": "None",
            }
        # Back-compat: ensure ground_truth_source exists
        external_metrics.setdefault("ground_truth_source", "TruthfulQA")
        external_risk = external_metrics["external_risk"]

        # Step 5: Final score
        print("\nStep 5: Computing final hybrid hallucination score...")
        final_risk = alpha * internal_risk_metrics["internal_risk"] + beta * external_risk
        print(f"  Final Hallucination Risk: {final_risk:.4f}")

        return {
            "prompt": prompt,
            "responses": responses,
            "primary_response": primary_generation["text"],
            "eigen": eigen_metrics,
            "stability": stability_metrics,
            "grounding": grounding_metrics,
            "internal_risk": internal_risk_metrics,
            "external": external_metrics,
            "final_risk": final_risk,
            "weights": {"alpha": alpha, "beta": beta, "w1": w1, "w2": w2, "w3": w3},
        }

    # (Ollama/Llama3 path removed - GPT-Neo is now the recommended larger model)
    
    def plot_eigenvalue_spectrum(self, eigenvalues: list, save_path: str = None):
        """
        Plot the eigenvalue spectrum for visualization.

        Args:
            eigenvalues: List of eigenvalues (descending order) from compute_eigen_score
            save_path: Optional path to save the plot
        """
        if not eigenvalues:
            raise ValueError("No eigenvalues were provided to plot.")

        ranked_eigenvalues = sorted((float(value) for value in eigenvalues), reverse=True)
        x_values = np.arange(1, len(ranked_eigenvalues) + 1)

        plt.figure(figsize=(12, 6))
        plt.bar(x_values, ranked_eigenvalues, color="steelblue", alpha=0.8)
        plt.plot(x_values, ranked_eigenvalues, color="#0f766e", marker="o", linewidth=2)
        plt.xlabel("Eigenvalue Rank", fontsize=12)
        plt.ylabel("Eigenvalue Magnitude", fontsize=12)
        plt.title("Hidden-State Covariance Eigenvalue Spectrum", fontsize=14, fontweight="bold")
        plt.xticks(x_values)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Eigenvalue spectrum saved to {save_path}")

        return plt
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a formatted summary of all results.
        
        Args:
            results: Results dictionary from analyze()
        """
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n--- GENERATED RESPONSES ---")
        for i, response in enumerate(results["responses"], 1):
            print(f"\nResponse {i}:")
            print(f"  {response}")
        
        print("\n--- INTERNAL METRICS ---")
        print(f"Eigen Score:      {results['eigen']['eigen_score']:.4f}")
        print(f"Responses used:   {results['eigen']['num_responses']}")
        print(f"Stability Score: {results['stability']['stability_score']:.4f}")
        print(f"Grounding Score: {results['grounding']['grounding_score']:.4f}")
        print(f"Internal Hallucination Risk: {results['internal_risk']['internal_risk']:.4f}")
        
        print("\n--- EXTERNAL METRICS ---")
        if results['external']['ground_truth'] != "N/A":
            print(f"Ground Truth: {results['external']['ground_truth']}")
            print("\nSimilarity Scores:")
            for i, sim in enumerate(results['external']['similarities'], 1):
                print(f"  Response {i}: {sim:.4f}")
            print(f"\nExternal Consistency: {results['external']['external_consistency']:.4f}")
            print(f"External Risk: {results['external']['external_risk']:.4f}")
        else:
            print("No ground truth available")
        
        print("\n--- FINAL SCORE ---")
        print(f"Final Hallucination Risk: {results['final_risk']:.4f}")
        
        # Risk interpretation
        risk = results['final_risk']
        if risk < 0.3:
            interpretation = "LOW - Response appears reliable"
        elif risk < 0.6:
            interpretation = "MEDIUM - Response may contain some uncertainties"
        else:
            interpretation = "HIGH - Response likely contains hallucinations"
        
        print(f"Risk Level: {interpretation}")
        print("="*80 + "\n")
