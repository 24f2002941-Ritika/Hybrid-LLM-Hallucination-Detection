"""
Model Loader Module.
Loads supported TransformerLens causal language models and provides text generation functionality.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Any
import numpy as np


class GPT2ModelLoader:
    """
    Handles loading and text generation with supported TransformerLens models.
    Supports: gpt2, gpt2-medium, gpt2-large, gpt2-xl,
              EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B,
              EleutherAI/pythia-2.8b, facebook/opt-6.7b
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the supported model variant to load
        """
        print(f"Loading {model_name} model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HookedTransformer.from_pretrained(model_name, device=self.device)
        self.model_name = model_name
        print(f"Model loaded successfully on {self.device}")
    
    def generate_responses(
        self, 
        prompt: str, 
        num_responses: int = 5,
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Generate multiple stochastic responses for a given prompt.
        
        Args:
            prompt: Input text prompt
            num_responses: Number of responses to generate
            max_length: Maximum length of generated text
            temperature: Sampling temperature for diversity
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generated text responses
        """
        responses = []
        
        # Compute prompt token length once
        prompt_tokens = self.model.to_tokens(prompt)
        prompt_token_len = prompt_tokens.shape[1]
        
        for i in range(num_responses):
            # Generate text
            generated_tokens = self.model.generate(
                prompt_tokens,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stop_at_eos=True
            )
            
            # Decode ONLY the newly generated tokens (not the prompt)
            new_tokens = generated_tokens[0][prompt_token_len:]
            generated_text = self.model.to_string(new_tokens).lstrip()
            responses.append(generated_text)
            
            print(f"Generated response {i+1}/{num_responses}")
        
        return responses
    
    def generate_with_cache(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate text and return both the text and model activations.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary containing generated text, tokens, logits, and cache
        """
        # Tokenize the prompt
        tokens = self.model.to_tokens(prompt)
        prompt_length = tokens.shape[1]
        
        # Generate with caching enabled
        with torch.no_grad():
            generated_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stop_at_eos=True,
                return_type="tokens"
            )
        
        # Get full sequence
        full_tokens = generated_tokens[0]
        
        # Run forward pass to get activations
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(full_tokens)
        
        # Decode ONLY the newly generated tokens (not the echoed prompt)
        new_tokens = full_tokens[prompt_length:]
        generated_text = self.model.to_string(new_tokens).lstrip()
        
        return {
            "text": generated_text,
            "tokens": full_tokens,
            "logits": logits,
            "cache": cache,
            "prompt_length": prompt_length
        }
    
    def get_model(self) -> HookedTransformer:
        """Return the underlying model."""
        return self.model
