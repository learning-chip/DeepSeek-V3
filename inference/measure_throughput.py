import os
import json

import torch
# "It is highly recommended to import intel_extension_for_pytorch right after import torch, prior to importing other packages."
# https://intel.github.io/intel-extension-for-pytorch/cpu/2.6.0+cpu/tutorials/getting_started.html
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from datasets import load_dataset
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from generate import sample, generate

from timeit import default_timer as timer

torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(8)  # NOTE: remember to change to available cores

def init_and_load_model(
    config_path,
    ckpt_path,
    device="cpu",
    max_batch_size=4,
    max_seq_len=256,
    use_ipex=True
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

    if use_ipex:
        # https://intel.github.io/intel-extension-for-pytorch/cpu/2.6.0+cpu/tutorials/getting_started.html#llm-quick-start
        model.eval()
        model = ipex.llm.optimize(
            model,
            dtype=torch.bfloat16,
            inplace=True,
            deployment_mode=True,
        )

    return args, model, tokenizer


def main(
    max_new_tokens=100,
    temperature=0.2,
    sample_offset=24
):

    ckpt_path = "/mount_home/DeepSeek-V2-Lite-Chat_converted"
    config_path = "./configs/config_16B.json"
    args, model, tokenizer = init_and_load_model(config_path, ckpt_path)

    ds = load_dataset("HuggingFaceH4/MATH-500")
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

    # warm-up
    generate(
        model,
        prompt_tokens,
        3,  # max_new_tokens
        tokenizer.eos_token_id,
        temperature
    )

    torch.manual_seed(0)
    start_time = timer()
    completion_tokens = generate(
        model,
        prompt_tokens,
        max_new_tokens,
        tokenizer.eos_token_id,
        temperature
    )
    end_time = timer()

    time_taken = end_time - start_time
    num_output_tokens = sum(len(output_sample) for output_sample in completion_tokens)

    print("num_output_tokens: ", num_output_tokens)
    print("time_taken (sec): ", time_taken)
    print("throughput (token/sec): ", num_output_tokens / time_taken)  
    # NOTE: here includes prefill time, but not counting prefill tokens

if __name__ == "__main__":
    main()
