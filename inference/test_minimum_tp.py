"""
MODEL_PATH_TP2=/scratch/model_weights/DeepSeek-V2-Lite-Chat_TP2
MODEL_CONFIG=configs/config_16B.json

CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node 2  test_minimum_tp.py \
    --ckpt-path $MODEL_PATH_TP2 --config $MODEL_CONFIG \
    2>&1 | tee print_tp_test.log
"""

import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from safetensors.torch import load_model

from model import Transformer, ModelArgs

def main(
    ckpt_path: str,
    config: str,
):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)

    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    print(
        f"[Rank {local_rank}] CUDA Memory Status: "
        f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, "
        f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, "
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB, "
        f"Max Reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB"
    )

    batch = 1
    num_tokens = 10  # NOTE: too large size leads to `Cuda failure 700 'an illegal memory access was encountered'` on BZ GPU
    tokens = torch.randint(0, args.vocab_size, size=(batch, num_tokens), dtype=torch.int64).to("cuda")
    output = model.forward(tokens)
    print(f"[rank {rank}] output.shape = {output.shape}")
    print(f"[rank {rank}] output.min = {output.min()}, output.max = {output.max()}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.ckpt_path, args.config)
