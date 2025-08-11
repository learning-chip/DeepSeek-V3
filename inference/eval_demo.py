"""
Dependency
    pip install transformers lm_eval==0.4.9.0 hqq==0.2.7.post1
    # 0.4.9.1 has API changes: https://github.com/EleutherAI/lm-evaluation-harness/releases

Usage
    WEIGHT_DIR=/workspace/model_weights/  # server-specific

    MODEL_PATH_TP1=$WEIGHT_DIR/DeepSeek-V2-Lite-Chat_TP1
    MODEL_CONFIG=configs/config_16B.json

    python eval_demo.py --ckpt-path $MODEL_PATH_TP1 --config $MODEL_CONFIG \
        --tasks gpqa_diamond_zeroshot | tee dsv2_minimumeval_gpqa_TP1.log

    python eval_demo.py --ckpt-path $MODEL_PATH_TP1 --config $MODEL_CONFIG \
        --tasks mmlu_high_school_computer_science mmlu_college_biology \
        | tee dsv2_minimumeval_mmlusubset_TP1.log

    python eval_demo.py --ckpt-path $MODEL_PATH_TP1 --config $MODEL_CONFIG \
        --tasks mmlu | tee dsv2_minimumeval_mmlu_TP1.log

    # quantize eval
    python eval_demo.py --ckpt-path $MODEL_PATH_TP1 --config $MODEL_CONFIG \
        --tasks mmlu_high_school_computer_science mmlu_college_biology \
        --quantize | tee dsv2_minimumeval_mmlusubset_TP1_hqqw4a16.log

Multi-device runs (if get error, check standalone dist.all_reduce on the GPU server)

    MODEL_PATH_TP2=$WEIGHT_DIR/DeepSeek-V2-Lite-Chat_TP2
    MODEL_CONFIG=configs/config_16B.json

    torchrun --standalone --nnodes 1 --nproc-per-node 2 \
        eval_demo.py --ckpt-path $MODEL_PATH_TP2 --config $MODEL_CONFIG \
        --tasks gpqa_diamond_zeroshot 2>&1 | tee dsv2_minimumeval_gpqa_TP2.log

    torchrun --standalone --nnodes 1 --nproc-per-node 2 \
        eval_demo.py --ckpt-path $MODEL_PATH_TP2 --config $MODEL_CONFIG \
        --tasks mmlu_high_school_computer_science mmlu_college_biology \
        | tee dsv2_minimumeval_mmlusubset_TP2.log

    # quantize eval
    torchrun --standalone --nnodes 1 --nproc-per-node 2 \
        eval_demo.py --ckpt-path $MODEL_PATH_TP2 --config $MODEL_CONFIG \
        --tasks mmlu_high_school_computer_science \
        --quantize | tee dsv2_minimumeval_mmlusubset_TP2_hqqw4a16.log

Weight preprocessing same as original generate.py

    # TP=1 preprocessing
    python ./convert.py \
        --hf-ckpt-path $WEIGHT_DIR/DeepSeek-V2-Lite-Chat \
        --save-path $WEIGHT_DIR/DeepSeek-V2-Lite-Chat_TP1 \
        --n-experts 64 --model-parallel 1
    # NOTE: V2-lite only has 64 routed experts

    # TP=2 preprocessing
    python ./convert.py \
        --hf-ckpt-path $WEIGHT_DIR/DeepSeek-V2-Lite-Chat \
        --save-path /workspace/model_weights/DeepSeek-V2-Lite-Chat_TP2 \
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


def report_memory(local_rank):
    # Due to torch.cuda.set_device call, will report on default device for each process
    print(
        f"[Rank {local_rank}] CUDA Memory Status: "
        f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, "
        f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, "
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB, "
        f"Max Reserved: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB"
    )


def main(args):
    ckpt_path = args.ckpt_path
    config = args.config
    tasks = args.tasks
    quantize = args.quantize

    if quantize:
        from model import Linear
        import hqq.models.base as hqq_base
        hqq_base._QUANT_LAYERS.append(Linear)  # let HQQ recognize custom Linear class
        print("_QUANT_LAYERS:", hqq_base._QUANT_LAYERS)

        from hqq.models.hf.base import AutoHQQHFModel  # also works for non-HF torch module
        from hqq.core.quantize import BaseQuantizeConfig
        from hqq.utils.patching import prepare_for_inference

        # TODO: use nanohqq for NPU backend

    # Configure for TP (`torchrun``), also works for TP=1 with simple `python` launch
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    if quantize:
        # NOTE: can also load to CPU and later quant on the fly to device, to reduce peak memory
        # but weight initialization on CPU is very slow
        with torch.device("cuda"):
            model = Transformer(args)

        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

        # quantize and pass to device
        quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
        model = AutoHQQHFModel.quantize_model(
            model,
            quant_config=quant_config,
            compute_dtype=torch.bfloat16,
            device="cuda",
            # skip_layer_pattern=None  # TODO: skip MoE gate layer
        )

        fast_backend = True
        if fast_backend:
            model.device = "cuda"
            model.dtype = torch.bfloat16
            prepare_for_inference(model, backend="torchao_int4") 
            # NOTE: torchao/gemlite backends are only fast for bs=1 (GEMV shape)
            # for batched inference (not used in eval here), need to adopt MARLIN backend
    else:
        with torch.device("cuda"):
            model = Transformer(args)

        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    report_memory(local_rank)
    
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
    print(f"[rank {rank}]", make_table(eval_results))

    report_memory(local_rank)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--tasks', nargs='+', type=str, default=["mmlu"], help='usage: --tasks task1 task2')
    parser.add_argument("--quantize", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
