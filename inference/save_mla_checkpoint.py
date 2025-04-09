"""
R1 evaluation data
- https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- https://huggingface.co/datasets/HuggingFaceH4/aime_2024
- https://huggingface.co/datasets/Idavidrein/gpqa
- https://huggingface.co/datasets/open-r1/codeforces

Full list:
https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#deepseek-r1-evaluation

Prepare model weights for this script:

    cd /home
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    apt install git-lfs
    git lfs install
    git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat

    # in this project dir
    cd inference
    pip install -r requirements.txt

    # single-device, no parallel split
    python ./convert.py \
        --hf-ckpt-path /home/DeepSeek-V2-Lite-Chat \
        --save-path /home/DeepSeek-V2-Lite-Chat_converted \
        --n-experts 64 --model-parallel 1  # V2-lite only has 64 routed experts
"""

import os
import json

import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from generate import sample, generate


torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(8)

dataset_list = [
    "HuggingFaceH4/MATH-500",
    "HuggingFaceH4/aime_2024",
    "open-r1/codeforces",
    "Idavidrein/gpqa"
]

def init_and_load_model(
    config_path,
    ckpt_path,
    device="cuda",
    max_batch_size=16,
    max_seq_len=256
):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    with open(config_path) as f:
        args = ModelArgs(**json.load(f))

    args.max_batch_size = max_batch_size  # for KV cache pre-allocation
    args.max_seq_len = max_seq_len

    with torch.device(device):
        model = Transformer(args)

    rank, world_size = 0, 1  # single-device
    load_model(
        model,
        os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    )
    return args, model, tokenizer


def main(
    output_dir="mla_ckpts",
    max_new_tokens=50,
    temperature=0.2,
    sample_offset=24
):

    ckpt_path = "/home/DeepSeek-V2-Lite-Chat_converted"
    config_path = "./configs/config_16B.json"
    args, model, tokenizer = init_and_load_model(config_path, ckpt_path)

    # TODO: loop over entire dataset
    ds = load_dataset(dataset_list[0])
    prompts = ds["test"]["problem"][sample_offset:sample_offset+args.max_batch_size]
    print("=== prompts: ===")
    for prompt in prompts:
        print(prompt)
        print("-"*10)

    prompt_tokens = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
            for prompt in prompts
        ]

    print("=== prompt_tokens length: ===")
    for prompt_token in prompt_tokens:
        print(len(prompt_token))

    # warm-up run and sanity check
    torch.manual_seed(0)
    completion_tokens = generate(
        model,
        prompt_tokens,
        20,  # max_new_tokens
        tokenizer.eos_token_id,
        temperature
    )

    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print("=" * 20)

    # configure checkpointing options
    for i, layer in enumerate(model.layers):
        current_dir = os.path.join(output_dir, f"layer{i:02}")
        os.makedirs(current_dir, exist_ok=True)

        mla_current = layer.attn
        mla_current.save_ckpt = True
        mla_current.ckpt_dir = current_dir
        mla_current.ckpt_iter = 0

    # long checkpointing run
    torch.manual_seed(0)
    _ = generate(
        model,
        prompt_tokens,
        max_new_tokens,
        tokenizer.eos_token_id,
        temperature
    )

if __name__ == "__main__":
    main()
