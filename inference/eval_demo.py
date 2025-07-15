"""
Usage

    MODEL_PATH=/scratch/model_weights/DeepSeek-V2-Lite-Chat_TP1
    MODEL_CONFIG=configs/config_16B.json

    python eval_demo.py --ckpt-path $MODEL_PATH --config $MODEL_CONFIG \
        --tasks gpqa_diamond_zeroshot | tee dsv2_minimumeval_gpqa.log

    python eval_demo.py --ckpt-path $MODEL_PATH --config $MODEL_CONFIG \
        --tasks mmlu_high_school_computer_science mmlu_college_biology \
        | tee dsv2_minimumeval_mmlusubset.log

    python eval_demo.py --ckpt-path $MODEL_PATH --config $MODEL_CONFIG \
        --tasks mmlu | tee dsv2_minimumeval_mmlu.log

# debug Tensor parallel

    MODEL_PATH_TP2=/scratch/model_weights/DeepSeek-V2-Lite-Chat_TP2
    MODEL_CONFIG=configs/config_16B.json

    NCCL_DEBUG=INFO CUDA_LAUNCH_BLOCKING=1 \
        torchrun --standalone --nnodes 1 --nproc-per-node 2 \
        eval_demo.py --ckpt-path $MODEL_PATH_TP2 --config $MODEL_CONFIG \
        --tasks gpqa_diamond_zeroshot 2>&1 | tee dsv2_minimumeval_gpqa_TP2.log

Weight preprocessing same as original generate.py

    # TP=1 preprocessing
    python ./convert.py \
        --hf-ckpt-path /scratch/model_weights/DeepSeek-V2-Lite-Chat \
        --save-path /scratch/model_weights/DeepSeek-V2-Lite-Chat_TP1 \
        --n-experts 64 --model-parallel 1
    # NOTE: V2-lite only has 64 routed experts

    # TP=2 preprocessing
    python ./convert.py \
        --hf-ckpt-path /scratch/model_weights/DeepSeek-V2-Lite-Chat \
        --save-path /scratch/model_weights/DeepSeek-V2-Lite-Chat_TP2 \
        --n-experts 64 --model-parallel 2
"""

import os
import json
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

import lm_eval
from lm_eval.utils import make_table

from model import Transformer, ModelArgs
from minimum_eval_wrapper import MinimumEvalWrapper


def main(
    ckpt_path: str,
    config: str,
    tasks: list = ["mmlu"]
) -> None:
    # Configure for TP (`torchrun``), also works for TP=1 with simple `python` launch
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")

    # NOTE: do not disable print for non-rank 0 processes
    # global print
    # if rank != 0:
    #     print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    lm = MinimumEvalWrapper(
        model,
        tokenizer
    )

    task_manager = lm_eval.tasks.TaskManager()
    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        task_manager=task_manager
    )
    print(make_table(eval_results))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--tasks', nargs='+', type=str, default=["mmlu"], help='usage: --tasks task1 task2')
    args = parser.parse_args()

    main(args.ckpt_path, args.config, args.tasks)
