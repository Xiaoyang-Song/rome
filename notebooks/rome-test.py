import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution
print(torch.cuda.is_available())

MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
print(model.config)

# OLD - From ROME paper
request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
]


# NEW - Customized for testing
request = [
    {
        "prompt": "{} equals to",
        "subject": "a + b",
        "target_new": {"str": "a^2 + b^2"},
    }
]

generation_prompts = [
    "What is the value of one plus one?",
    "2 + 2 equals?",
    "Jerry bought two oranges and one bananas yesterday. How many fruits did he purchase in total?",
    "Does 1 + 10 equal to 11?",
    "Two oranges plus three oranges equal to",
]


ALG_NAME = "ROME"

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")


# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME
)

stop_execution()

generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True)

request = [
    {
        "prompt": "{} plays the sport of",
        "subject": "LeBron James",
        "target_new": {"str": "football"},
    }
]

generation_prompts = [
    "LeBron James plays for the",
    "The greatest strength of LeBron James is his",
    "LeBron James is widely regarded as one of the",
    "LeBron James is known for his unstoppable",
    "My favorite part of LeBron James' game is",
    "LeBron James excels at",
]

request = [
    {
        "prompt": "{} was developed by",
        "subject": "Mario Kart",
        "target_new": {
            "str": "Apple",
        },
    }
]

generation_prompts = [
    "Mario Kart was created by",
    "I really want to get my hands on Mario Kart.",
    "Mario Kart is",
    "Which company created Mario Kart?",
]