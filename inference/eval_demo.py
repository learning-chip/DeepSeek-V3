"""
MODEL_PATH=/scratch/model_weights/DeepSeek-V2-Lite-Chat_TP1
MODEL_CONFIG=configs/config_16B.json

python eval_demo.py --ckpt-path $MODEL_PATH --config $MODEL_CONFIG \
    --tasks gpqa_diamond_zeroshot | tee dsv2_minimumeval_gpqa.log

python eval_demo.py --ckpt-path $MODEL_PATH --config $MODEL_CONFIG \
    --tasks mmlu_high_school_computer_science mmlu_college_biology \
    | tee dsv2_minimumeval_mmlusubset.log

python eval_demo.py --ckpt-path $MODEL_PATH --config $MODEL_CONFIG \
    --tasks mmlu
"""

import os
import json
from argparse import ArgumentParser

import torch
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
    # TODO: add TP
    torch.cuda.set_device("cuda:0")
    world_size = 1
    rank = 0

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
    return eval_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--tasks', nargs='+', type=str, default=["mmlu"], help='usage: --tasks task1 task2')
    args = parser.parse_args()

    main(args.ckpt_path, args.config, args.tasks)
