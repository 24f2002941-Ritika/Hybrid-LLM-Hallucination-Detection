"""
Internal Metrics Module
Computes eigen score, stability, and attention grounding metrics
for hallucination detection.

Eigen Score follows the INSIDE paper (ICLR 2024): it measures semantic
consistency across multiple sampled responses by analysing the eigenvalue
spectrum of the covariance matrix built from per-response sentence embeddings
extracted at the middle Transformer layer.  A low (more negative) eigen score
signals high semantic consistency (low hallucination risk); a high (less
negative) score signals high dispersion (uncertain / likely hallucinated).

Feature Clipping follows the INSIDE paper exactly:
  - A memory bank M of shape (N, hidden_dim) accumulates token-level hidden
    activations across forward passes.
  - Per-dimension thresholds are derived from the memory bank:
        h_min[j] = percentile(M[:, j], 0.2)
        h_max[j] = percentile(M[:, j], 99.8)
  - Every hidden-state tensor is clipped element-wise per feature dimension j
    *before* the sentence embedding is extracted.
"""
#hiiiiis
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Memory Bank
# ---------------------------------------------------------------------------

class MemoryBank:
    """
    Accumulates token-level hidden-state vectors and provides
    distribution-aware, per-dimension feature clipping as described in the
    INSIDE paper (ICLR 2024).

    Usage
    -----
    1. Call ``update(hidden)`` after every forward pass to feed new
       activations into the bank.
    2. Call ``clip(hidden)`` to apply the derived thresholds to any
       hidden-state tensor before downstream computation.

    Attributes
    ----------
    max_size : int
        Maximum number of activation vectors retained (FIFO when full).
    hidden_dim : int or None
        Dimensionality of the stored vectors (inferred on first update).
    h_min, h_max : torch.Tensor or None
        Per-dimension clipping thresholds. None until the bank has been
        populated and ``compute_thresholds()`` has been called.
    """

    def __init__(self, max_size: int = 2000, lo: float = 0.2, hi: float = 99.8):
        """
        Args:
            max_size: Maximum stored activation samples N.
            lo:       Lower percentile for h_min  (paper: 0.2).
            hi:       Upper percentile for h_max  (paper: 99.8).
        """
        self.max_size = max_size
        self.lo = lo
        self.hi = hi

        self._bank: List[torch.Tensor] = []   # list of 1-D float32 CPU tensors
        self.hidden_dim: Optional[int] = None
        self.h_min: Optional[torch.Tensor] = None
        self.h_max: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, hidden: torch.Tensor) -> None:
        """
        Add token-level activation vectors from a hidden-state tensor to
        the memory bank.

        Args:
            hidden: Tensor of shape (seq_len, hidden_dim) **or**
                    (batch, seq_len, hidden_dim).  All token vectors are
                    flattened to individual rows and appended to the bank.
        """
        h = hidden.detach().float().cpu()

        if h.dim() == 3:           # (batch, seq_len, hidden_dim)
            h = h.reshape(-1, h.size(-1))
        elif h.dim() == 2:         # (seq_len, hidden_dim) — already flat
            pass
        else:
            raise ValueError(f"Expected 2-D or 3-D hidden tensor, got shape {tuple(h.shape)}")

        if self.hidden_dim is None:
            self.hidden_dim = h.size(-1)

        # Append individual row vectors
        for vec in h:
            self._bank.append(vec)

        # Enforce FIFO capacity
        if len(self._bank) > self.max_size:
            self._bank = self._bank[-self.max_size:]

        # Invalidate cached thresholds — they need recomputing
        self.h_min = None
        self.h_max = None

    def compute_thresholds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-dimension clipping thresholds from the current bank.

        Returns:
            (h_min, h_max) — 1-D tensors of shape (hidden_dim,).

        Raises:
            RuntimeError: If the bank is empty.
        """
        if len(self._bank) == 0:
            raise RuntimeError(
                "MemoryBank is empty. Call update() with hidden activations first."
            )

        # Stack to (N, hidden_dim) — stays on CPU for numpy percentile
        M = torch.stack(self._bank).numpy()          # (N, hidden_dim)

        # Per-dimension percentiles (axis=0 → operate along the N dimension)
        h_min_np = np.percentile(M, self.lo, axis=0)   # (hidden_dim,)
        h_max_np = np.percentile(M, self.hi, axis=0)   # (hidden_dim,)

        self.h_min = torch.from_numpy(h_min_np.astype(np.float32))
        self.h_max = torch.from_numpy(h_max_np.astype(np.float32))
        return self.h_min, self.h_max

    def clip(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply distribution-aware, per-dimension feature clipping to *hidden*.

        The thresholds are (re)computed from the current memory bank the
        first time this is called, or whenever the bank has been updated
        since the last call.

        Args:
            hidden: Tensor of shape (seq_len, hidden_dim) **or**
                    (batch, seq_len, hidden_dim).

        Returns:
            Clipped tensor with the same shape and device as *hidden*.
        """
        if len(self._bank) == 0:
            # Nothing in the bank yet — skip clipping silently
            return hidden

        if self.h_min is None or self.h_max is None:
            self.compute_thresholds()

        # Move thresholds to the same device as the incoming tensor
        h_min = self.h_min.to(hidden.device)   # (hidden_dim,)
        h_max = self.h_max.to(hidden.device)   # (hidden_dim,)

        # torch.clamp with per-element min/max tensors — works on any shape
        # because h_min/h_max broadcast over all leading dimensions.
        return torch.clamp(hidden, min=h_min, max=h_max)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._bank)

    def __repr__(self) -> str:
        return (
            f"MemoryBank(size={len(self._bank)}/{self.max_size}, "
            f"hidden_dim={self.hidden_dim}, "
            f"percentiles=[{self.lo}, {self.hi}])"
        )


# ---------------------------------------------------------------------------
# Internal Metrics
# ---------------------------------------------------------------------------

class InternalMetrics:
    """
    Computes internal hallucination metrics from model activations.
    """

    def __init__(self, model, memory_bank_size: int = 2000):
        """
        Initialize with a TransformerLens model.

        Args:
            model:            HookedTransformer model instance.
            memory_bank_size: Capacity of the feature-clipping memory bank.
        """
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads

        # One memory bank per InternalMetrics instance, shared across calls.
        # Tracks hidden states at the middle layer (consistent with EigenScore).
        self.memory_bank = MemoryBank(max_size=memory_bank_size, lo=0.2, hi=99.8)

    # ------------------------------------------------------------------
    # EigenScore (INSIDE, ICLR 2024) with feature clipping
    # ------------------------------------------------------------------

    def compute_eigen_score(
        self,
        responses: List[str],
        alpha: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Compute EigenScore as defined in the INSIDE paper with per-dimension
        feature clipping applied to middle-layer hidden states.

        Pipeline
        --------
        For each of the K sampled responses:
          1. Run forward pass, extract middle-layer residual stream:
                 H_i  ∈  R^(seq_len × d_model)
          2. Feed H_i into the memory bank (updates statistics).
          3. Apply distribution-aware, per-dimension feature clipping:
                 H_i_clipped = clip(H_i, h_min, h_max)
             where h_min[j] = percentile(M[:,j], 0.2)
                   h_max[j] = percentile(M[:,j], 99.8)
             and M is built from all activations accumulated so far.
          4. Extract sentence embedding from last token of clipped states:
                 z_i = H_i_clipped[-1, :]   ∈  R^d

        Build embedding matrix:
             Z = stack([z_1, …, z_K])                       shape: (K, d)
        Centre and form the (K×K) covariance:
             Z_centred = Z - Z.mean(dim=0)
             Σ = Z_centred @ Z_centred.T + α·I              shape: (K, K)
        Eigendecompose Σ (symmetric → eigvalsh):
             λ_1 … λ_K  (clamped to ≥ 1e-12)
        EigenScore = (1/K) · Σ_i log(λ_i)

        Interpretation
        --------------
        Lower (more negative) → tightly clustered embeddings → low hallucination.
        Higher (less negative / positive) → dispersed embeddings → likely hallucinated.

        Args:
            responses: List of K response strings.
            alpha:     Tikhonov regularisation coefficient (default 0.001).

        Returns:
            Dictionary with:
                eigen_score       – (1/K) · Σ log(λ_i)
                eigenvalues       – list of eigenvalues in ascending order
                num_responses     – K
                clipping_applied  – True if the memory bank had enough data
                h_min             – per-dimension lower thresholds (list, for debug)
                h_max             – per-dimension upper thresholds (list, for debug)
        """
        K = len(responses)
        if K == 0:
            return {
                "eigen_score": 0.0,
                "eigenvalues": [],
                "num_responses": 0,
                "clipping_applied": False,
                "h_min": [],
                "h_max": [],
            }

        mid_layer = self.n_layers // 2
        resid_key = f"blocks.{mid_layer}.hook_resid_post"

        # ------------------------------------------------------------------
        # PASS 1: collect all middle-layer hidden states into the memory bank
        # ------------------------------------------------------------------
        raw_hiddens: List[torch.Tensor] = []   # (seq_len, d_model) per response

        for response in responses:
            tokens = self.model.to_tokens(response)          # [1, seq_len]
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)

            # hidden: (seq_len, d_model), float32
            hidden = cache[resid_key][0].float()
            raw_hiddens.append(hidden)

            # Feed into memory bank to build distribution statistics
            self.memory_bank.update(hidden)

        clipping_applied = len(self.memory_bank) > 0

        # Retrieve the thresholds computed from ALL K responses (and any
        # activations accumulated from previous calls).
        if clipping_applied:
            h_min, h_max = self.memory_bank.compute_thresholds()
        else:
            h_min = h_max = None

        # ------------------------------------------------------------------
        # PASS 2: clip each hidden state, extract sentence embedding
        # ------------------------------------------------------------------
        embeddings: List[torch.Tensor] = []

        for hidden in raw_hiddens:
            # Apply INSIDE-paper feature clipping (per-dimension, from bank)
            hidden_clipped = self.memory_bank.clip(hidden)        # (seq_len, d_model)

            # Last-token embedding (INSIDE paper §3.2)
            z_i = hidden_clipped[-1, :]                           # (d_model,)

            # L2-normalise onto the unit sphere so that eigenvalues measure
            # *angular* (directional) dispersion rather than magnitude.
            # Without this, GPT-2 hidden norms (~35-50) inflate eigenvalues
            # by ~1000×, pushing the eigen score to large positive numbers.
            z_i = F.normalize(z_i, p=2, dim=0)                   # unit vector
            embeddings.append(z_i)

        # ------------------------------------------------------------------
        # EigenScore computation
        # ------------------------------------------------------------------
        Z = torch.stack(embeddings)                               # (K, d_model)

        Z_mean = Z.mean(dim=0)
        Z_centered = Z - Z_mean                                   # (K, d_model)

        # (K × K) covariance with Tikhonov regularisation
        # With unit-norm embeddings, Sigma entries are bounded in [-1, 1]
        # and eigenvalues lie in [0, K], giving log(λ) a sensible range.
        Sigma = Z_centered @ Z_centered.T                         # (K, K)
        Sigma = Sigma + alpha * torch.eye(K, device=Sigma.device, dtype=Sigma.dtype)

        # Eigenvalues (symmetric → eigvalsh gives real, ascending values)
        try:
            eigenvalues = torch.linalg.eigvalsh(Sigma)
        except Exception:
            eigenvalues = torch.abs(torch.linalg.eigvals(Sigma).real)

        eigenvalues = torch.clamp(eigenvalues, min=1e-12)

        # EigenScore: (1/K) * sum(log(λ_i))
        eigen_score = float((torch.sum(torch.log(eigenvalues)) / K).cpu())

        return {
            "eigen_score": eigen_score,
            "eigenvalues": eigenvalues.cpu().tolist(),
            "num_responses": K,
            "clipping_applied": clipping_applied,
            "h_min": h_min.tolist() if h_min is not None else [],
            "h_max": h_max.tolist() if h_max is not None else [],
        }

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def compute_stability(self, cache: Dict, prompt_length: int) -> Dict[str, float]:
        """
        Compute stability metric based on hidden state similarity across layers.

        Args:
            cache:         Model activation cache.
            prompt_length: Length of the prompt in tokens.

        Returns:
            Dictionary with stability score.
        """
        layer_activations = []

        for layer_idx in range(self.n_layers):
            resid_key = f"blocks.{layer_idx}.hook_resid_post"
            if resid_key in cache:
                activation = cache[resid_key]             # [batch, seq_len, d_model]
                layer_activations.append(activation[0])   # first batch item

        if len(layer_activations) < 2:
            return {"stability_score": 1.0}

        similarities = []

        for i in range(len(layer_activations) - 1):
            curr_layer = layer_activations[i]             # [seq_len, d_model]
            next_layer = layer_activations[i + 1]

            curr_norm = F.normalize(curr_layer, p=2, dim=-1)
            next_norm = F.normalize(next_layer, p=2, dim=-1)

            token_similarities = torch.sum(curr_norm * next_norm, dim=-1)  # [seq_len]
            similarities.append(token_similarities)

        all_similarities = torch.stack(similarities)      # [n_layers-1, seq_len]
        stability_score = float(torch.mean(all_similarities).cpu().numpy())

        return {
            "stability_score": stability_score,
            "layer_similarities": [float(torch.mean(s).cpu().numpy()) for s in similarities],
        }

    # ------------------------------------------------------------------
    # Attention Grounding
    # ------------------------------------------------------------------

    def compute_attention_grounding(
        self,
        cache: Dict,
        prompt_length: int,
        total_length: int,
    ) -> Dict[str, float]:
        """
        Compute attention grounding — how much attention generated tokens pay
        to prompt tokens.

        Args:
            cache:         Model activation cache.
            prompt_length: Length of the prompt in tokens.
            total_length:  Total sequence length.

        Returns:
            Dictionary with grounding score.
        """
        grounding_scores = []

        for layer_idx in range(self.n_layers):
            attn_key = f"blocks.{layer_idx}.attn.hook_pattern"

            if attn_key in cache:
                attn_pattern = cache[attn_key]       # [batch, n_heads, seq_len, seq_len]
                attn_pattern = attn_pattern[0]       # [n_heads, seq_len, seq_len]

                if total_length > prompt_length:
                    gen_attn = attn_pattern[:, prompt_length:, :]        # [n_heads, gen_len, seq_len]
                    attn_to_prompt = torch.sum(gen_attn[:, :, :prompt_length], dim=-1)
                    total_attn = torch.sum(gen_attn, dim=-1)

                    grounding_ratio = attn_to_prompt / (total_attn + 1e-10)
                    layer_grounding = float(torch.mean(grounding_ratio).cpu().numpy())
                    grounding_scores.append(layer_grounding)

        if len(grounding_scores) == 0:
            return {"grounding_score": 1.0}

        grounding_score = float(np.mean(grounding_scores))

        return {
            "grounding_score": grounding_score,
            "layer_grounding": grounding_scores,
        }

    # ------------------------------------------------------------------
    # Internal Risk
    # ------------------------------------------------------------------

    def compute_internal_risk(
        self,
        eigen_metrics: Dict,
        stability_metrics: Dict,
        grounding_metrics: Dict,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.3,
    ) -> Dict[str, float]:
        """
        Compute internal hallucination risk score.

        Args:
            eigen_metrics:    Output from compute_eigen_score.
            stability_metrics: Output from compute_stability.
            grounding_metrics: Output from compute_attention_grounding.
            w1, w2, w3:       Weights for eigen score, stability, grounding.

        Returns:
            Dictionary with internal risk score and components.
        """
        import math

        raw = eigen_metrics["eigen_score"]
        # Map raw eigen score to [0, 1] risk.
        # With L2-normalised embeddings the raw score typically falls in
        # [-2, +2].  A simple sigmoid centred at 0 with scale=1 works well:
        #   score << 0  → tight cluster   → low risk  (→ 0)
        #   score ~  0  → moderate spread → mid risk  (→ 0.5)
        #   score >> 0  → high dispersion → high risk (→ 1)
        normalized_eigen = float(1.0 / (1.0 + math.exp(-raw)))

        stability = stability_metrics["stability_score"]
        grounding = grounding_metrics["grounding_score"]

        internal_risk = (
            w1 * normalized_eigen +
            w2 * (1 - stability) +
            w3 * (1 - grounding)
        )

        return {
            "internal_risk": internal_risk,
            "eigen_score_component": w1 * normalized_eigen,
            "stability_component": w2 * (1 - stability),
            "grounding_component": w3 * (1 - grounding),
        }
