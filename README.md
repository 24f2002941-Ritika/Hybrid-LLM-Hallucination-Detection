---
title: Inside the Mind of an LLM
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Hybrid LLM Hallucination Detection System

A complete Python project for detecting hallucinations in LLM-generated text using GPT-2 and TransformerLens. The system combines internal model analysis (entropy, stability, attention grounding) with external factual verification.

## Features

- **Internal Hallucination Analysis**
  - Token-level entropy computation
  - Layer-wise stability analysis via hidden state similarity
  - Attention grounding metric (attention to prompt tokens)
  
- **External Factual Verification**
  - Ground truth dataset matching
  - Semantic similarity using sentence transformers
  - Multi-response consistency checking

- **Hybrid Risk Scoring**
  - Weighted combination of internal and external metrics
  - Configurable risk weights
  - Interpretable risk levels (LOW/MEDIUM/HIGH)

- **Interactive Streamlit UI**
  - Real-time analysis dashboard
  - Interactive visualizations with Plotly
  - Configurable parameters
  - Comprehensive metric displays

## Project Structure

```
gllm/
├── model_loader.py          # GPT-2 model loading and text generation
├── internal_metrics.py      # Entropy, stability, grounding metrics
├── external_verifier.py     # Ground truth matching and similarity
├── analyzer.py              # Main analysis orchestrator
├── main.py                  # CLI interface
├── app.py                   # Streamlit web UI
├── ground_truth.json        # Sample ground truth dataset
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Streamlit Web UI (Recommended)

Run the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser to the provided URL (typically http://localhost:8501)

Features:
- Enter prompts and configure parameters via UI
- View all metrics and visualizations in real-time
- Interactive plots for entropy curves and metric comparisons
- Export-ready analysis results

### Option 2: Command Line Interface

Run analysis from the command line:

```bash
python main.py --prompt "What is the capital of France?" --num-responses 5
```

Available arguments:
- `--prompt`: Input question/prompt (required)
- `--num-responses`: Number of responses to generate (default: 5)
- `--max-length`: Maximum generation length (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
- `--model`: GPT-2 variant (default: gpt2)
- `--ground-truth`: Path to ground truth JSON (default: ground_truth.json)
- `--save-plot`: Path to save entropy plot (optional)
- `--alpha`: Weight for internal risk (default: 0.6)
- `--beta`: Weight for external risk (default: 0.4)

Example:
```bash
python main.py --prompt "Who wrote Romeo and Juliet?" --num-responses 5 --temperature 0.9 --save-plot entropy.png
```

## How It Works

### 1. Text Generation
- Loads GPT-2 using TransformerLens
- Generates 5 stochastic responses with temperature sampling
- Captures model activations (logits, hidden states, attention patterns)

### 2. Internal Metrics

**Entropy Metric:**
- Computes token-level entropy from logits
- Higher entropy = higher uncertainty

**Stability Metric:**
- Measures cosine similarity between consecutive layer activations
- Lower stability = more processing changes (potential hallucination)

**Attention Grounding:**
- Ratio of attention to prompt tokens vs all tokens
- Lower grounding = less reliance on input (potential hallucination)

**Internal Risk Score:**
```
InternalRisk = w1 * entropy + w2 * (1 - stability) + w3 * (1 - grounding)
```

### 3. External Verification

- Matches prompt to ground truth dataset
- Computes semantic similarity using sentence-transformers
- Averages similarity across all 5 responses
- External risk = 1 - consistency

### 4. Final Hybrid Score

```
FinalRisk = alpha * InternalRisk + beta * ExternalRisk
```

Default: alpha=0.6, beta=0.4

### 5. Risk Interpretation

- **< 0.3**: LOW - Response appears reliable
- **0.3-0.6**: MEDIUM - May contain uncertainties
- **> 0.6**: HIGH - Likely contains hallucinations

## Customization

### Adding Ground Truth Data

Edit `ground_truth.json`:

```json
[
    {
        "question": "Your question here",
        "answer": "The factual answer here"
    }
]
```

### Adjusting Weights

In the Streamlit UI, use the sidebar sliders.

For CLI, use command-line arguments:
```bash
python main.py --prompt "..." --alpha 0.7 --beta 0.3
```

### Using Different Models

The system supports GPT-2 variants:
- `gpt2` (124M parameters)
- `gpt2-medium` (355M parameters)
- `gpt2-large` (774M parameters)

```bash
python main.py --prompt "..." --model gpt2-medium
```

## Output

The system provides:

1. **All 5 generated responses**
2. **Internal metrics:**
   - Mean and max entropy
   - Stability score
   - Grounding score
   - Internal risk score
3. **External metrics:**
   - Similarity scores for each response
   - External consistency
   - External risk
4. **Final hallucination risk score**
5. **Visualizations:**
   - Entropy curve plot
   - Metric comparison charts
   - Risk breakdown

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TransformerLens
- Sentence Transformers
- Streamlit
- Matplotlib/Plotly

See `requirements.txt` for complete list.

## Notes

- First run will download GPT-2 and sentence-transformer models (~500MB total)
- GPU recommended but not required (runs on CPU)
- Analysis takes ~30-60 seconds per prompt on CPU

## License

MIT License - feel free to use and modify for your projects.

## Citation

If you use this system in your research, please cite:

```
Hybrid LLM Hallucination Detection System
Using GPT-2 and TransformerLens
https://github.com/yourusername/gllm
```
