"""
Random In vs Random Out: Latent Space Perturbation Experiment

Two approaches to introducing diversity in LLM outputs:
  - Random In  : Inject Gaussian noise into token embeddings before inference (temp=0)
  - Random Out : Standard temperature sampling during token generation

We test whether front-loaded noise (Random In) produces more coherent
diversity than step-by-step randomness (Random Out).

Model  : Qwen/Qwen2.5-7B-Instruct
Device : CUDA (tested on Kaggle T4 x2)
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {device}")


# Model
class LatentExperimentRunner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"[Init] Loading {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        print("[Init] Model ready.\n")

    # Random In
    def generate_latent_perturbation(
        self,
        prompt: str,
        noise_scale: float = 0.0,
        max_new_tokens: int = 256
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

        if noise_scale > 0.0:
            noise = torch.randn_like(input_embeds) * noise_scale
            input_embeds = input_embeds + noise

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Random Out
    def generate_temp_baseline(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 256
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Evaluation Problems
EVALUATION_SUITE = [
    {
        "id": "logic_reasoning",
        "category": "Logic",
        "prompt": (
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
            "The boat holds only the farmer and one item. Solve step-by-step."
        )
    },
    {
        "id": "code_optimization",
        "category": "Code",
        "prompt": (
            "Write a Python function to find the maximum sub-array sum with dynamic "
            "programming. Explain the time and space complexity."
        )
    },
    {
        "id": "system_design",
        "category": "Architecture",
        "prompt": (
            "Propose a fault-tolerant architecture for a high-throughput messaging "
            "queue handling 1M writes/sec."
        )
    },
    {
        "id": "math_derivation",
        "category": "Math",
        "prompt": (
            "Derive the closed-form expression for the sum of the first N squares "
            "step-by-step."
        )
    },
    {
        "id": "strategy_creative",
        "category": "Strategy",
        "prompt": (
            "Suggest three distinct, non-obvious ways to reduce urban heat islands "
            "in mega-cities without using traditional urban forestry."
        )
    }
]

# Prompt template
def format_prompt(problem_text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a precise reasoning model."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{problem_text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Main Pipeline
if __name__ == "__main__":
    runner = LatentExperimentRunner()

    # Noise scales: 0.0 is the clean deterministic baseline
    noise_scales       = [0.0, 0.02, 0.05, 0.1, 0.2]
    num_trials         = 3    # trials per noise scale (for diversity measurement)

    results = []

    print("[Running Experiment Suite]...\n")

    for problem in EVALUATION_SUITE:
        print(f" Category: {problem['category']} ({problem['id']})")
        prompt = format_prompt(problem["prompt"])

        # 1. Random Out baseline - temperature sampling
        print("   [Random Out] Temperature sampling @ 0.7 ...")
        out = runner.generate_temp_baseline(prompt, temperature=0.7)
        results.append({
            "problem_id"  : problem["id"],
            "category"    : problem["category"],
            "method"      : "Random_Out_Temp0.7",
            "noise_scale" : 0.70,
            "trial"       : 1,
            "output"      : out
        })

        # 2. Random In - latent perturbation across noise scales
        for scale in noise_scales:
            for trial in range(1, num_trials + 1):
                out = runner.generate_latent_perturbation(prompt, noise_scale=scale)
                results.append({
                    "problem_id"  : problem["id"],
                    "category"    : problem["category"],
                    "method"      : "Random_In_Latent",
                    "noise_scale" : scale,
                    "trial"       : trial,
                    "output"      : out
                })
                print(f"   [Random In]  scale={scale} | trial={trial} done")

        print()

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/latent_perturbation_results.csv", index=False)
    print("[Done] Results saved to results/latent_perturbation_results.csv")
