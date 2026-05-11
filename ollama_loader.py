"""
Ollama Model Loader
Handles text generation via the local Ollama API (e.g., llama3).
Because Ollama models don't expose internal logits/activations,
internal metrics are approximated from response-level statistics.
"""

import requests
import json
import numpy as np
from typing import List, Dict, Any


OLLAMA_BASE_URL = "http://localhost:11434"


def _check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _resolve_model_name(requested: str) -> str:
    """
    Look up the exact tag Ollama knows for the requested model.
    Returns the resolved tag (e.g. 'llama3:latest') or the original
    string if nothing matches (Ollama may still handle it).
    """
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return requested
        models = r.json().get("models", [])
        tags = [m["name"] for m in models]

        # Exact match first
        if requested in tags:
            return requested

        # Prefix match: 'llama3' matches 'llama3:latest', 'llama3:8b', etc.
        base = requested.split(":")[0]
        matches = [t for t in tags if t == requested or t.startswith(base + ":")]
        if matches:
            return matches[0]  # prefer first match (usually :latest)

        return requested  # let Ollama decide / fail with a clear message
    except Exception:
        return requested


class OllamaModelLoader:
    """
    Loads text generation capability from a locally running Ollama model.
    Internal metrics (entropy, stability, grounding) are approximated because
    Ollama does not expose raw logits or attention weights via its public API.
    """

    def __init__(self, model_name: str = "llama3"):
        if not _check_ollama_running():
            raise RuntimeError(
                "Ollama server is not running. "
                "Please start it with: `ollama serve`"
            )
        # Resolve to the exact tag Ollama knows (e.g. 'llama3' → 'llama3:latest')
        self.model_name = _resolve_model_name(model_name)
        self.api_url = f"{OLLAMA_BASE_URL}/api/generate"
        print(f"Ollama model resolved to '{self.model_name}' at {OLLAMA_BASE_URL}")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate_single(
        self,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.8,
    ) -> str:
        """Call Ollama API and return the generated text (non-streaming)."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_length),
            },
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            if not response.ok:
                # Surface Ollama's own error message for easier debugging
                try:
                    detail = response.json().get("error", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(
                    f"Ollama API error ({response.status_code}): {detail}\n"
                    f"Model used: '{self.model_name}'"
                )
            return response.json().get("response", "")
        except requests.RequestException as e:
            raise RuntimeError(f"Request to Ollama failed: {e}") from e

    def generate_responses(
        self,
        prompt: str,
        num_responses: int = 5,
        max_length: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,  # accepted for API compatibility, not forwarded
    ) -> List[str]:
        """Generate multiple stochastic responses for a given prompt."""
        responses = []
        for i in range(num_responses):
            text = self._generate_single(prompt, max_length, temperature)
            responses.append(text)
            print(f"Generated response {i + 1}/{num_responses}")
        return responses

    # ------------------------------------------------------------------
    # Proxy internal metrics
    # Because Ollama doesn't expose logits/activations, we approximate
    # entropy from token-probability variance and stability from the
    # pairwise similarity of multiple responses.
    # ------------------------------------------------------------------

    def generate_with_proxy_metrics(
        self,
        prompt: str,
        num_samples: int = 5,
        max_length: int = 150,
        temperature: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Generate several responses and derive proxy internal-metric signals.

        Returns a dict with keys mirroring those expected by analyzer.py:
          - responses          : list of generated strings
          - entropy_metrics    : dict with mean_entropy, max_entropy, entropy_curve
          - stability_metrics  : dict with stability_score, layer_similarities
          - grounding_metrics  : dict with grounding_score
        """
        responses = self.generate_responses(
            prompt, num_responses=num_samples,
            max_length=max_length, temperature=temperature
        )

        # --- Proxy entropy: estimated from character-length variance ----
        # High variance in response length / token counts ≈ high uncertainty.
        lengths = np.array([len(r.split()) for r in responses], dtype=float)
        if lengths.max() > 0:
            normalised = lengths / lengths.max()
        else:
            normalised = lengths
        # Build a synthetic entropy curve (one value per response)
        prob_like = normalised / (normalised.sum() + 1e-9)
        entropy_curve = list(
            -prob_like * np.log2(prob_like + 1e-9)
        )
        mean_entropy = float(np.mean(entropy_curve))
        max_entropy  = float(np.max(entropy_curve))

        # --- Proxy stability: Jaccard overlap between response pairs -----
        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            if not sa and not sb:
                return 1.0
            return len(sa & sb) / len(sa | sb)

        pair_sims = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pair_sims.append(jaccard(responses[i], responses[j]))
        stability_score = float(np.mean(pair_sims)) if pair_sims else 0.5
        # Expose as layer-like list so the plot in app.py doesn't break
        layer_similarities = pair_sims if pair_sims else [stability_score]

        # --- Proxy grounding: similarity of responses to the prompt -----
        prompt_words = set(prompt.lower().split())
        grounding_scores = []
        for r in responses:
            r_words = set(r.lower().split())
            if not r_words:
                grounding_scores.append(0.0)
            else:
                grounding_scores.append(len(prompt_words & r_words) / len(r_words))
        grounding_score = float(np.mean(grounding_scores))

        return {
            "responses": responses,
            "entropy_metrics": {
                "mean_entropy": mean_entropy,
                "max_entropy":  max_entropy,
                "entropy_curve": entropy_curve,
            },
            "stability_metrics": {
                "stability_score":    stability_score,
                "layer_similarities": layer_similarities,
            },
            "grounding_metrics": {
                "grounding_score": grounding_score,
            },
        }
