"""
Dependency
    pip install transformers==4.55.0 lm_eval==0.4.9.0

Usage

    # multiple choice
    python hf_lm_eval.py --tasks mmlu | tee dsv2_hfref_eval_mmlu.log

    # PPL
    python hf_lm_eval.py --tasks wikitext | tee dsv2_hfref_eval_wikitext.log
    python hf_lm_eval.py --tasks pile_10k | tee dsv2_hfref_eval_pile10k.log
    python hf_lm_eval.py --tasks pile_arxiv | tee dsv2_hfref_eval_arxiv.log
    # NOTE: full `pile` or `c4` datasets are hundreds of GBs, here only look at small subsets

Limitation:
    No robust way to use TP/EP with HF DS, better use manual (non-HF) model.py to scale to multiple devices.
"""

from timeit import default_timer as timer
import argparse
import torch

import lm_eval
from lm_eval import evaluator
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

from transformers import AutoTokenizer
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM
# built-in DS model added since transformers==4.54.0
# https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py


def main(args):
    print("args: ", args)
    model_path = args.model_path
    device = args.device
    torch.cuda.set_device(device)

    model = DeepseekV2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = HFLM(
        pretrained=model,
        tokenizer=tokenizer
    )
    # TODO: can add quantization here

    start_time = timer()
    results = evaluator.simple_evaluate(
        model=model,
        tasks=args.tasks,
        task_manager=lm_eval.tasks.TaskManager(),
        batch_size=args.batch_size,
        write_out=True,
        log_samples=False
    )

    eval_time = timer() - start_time
    print(f"Evaluation took {eval_time:.2f} seconds")

    # Print results
    print(make_table(results))
    print(results["results"].keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str,
        default="/scratch/model_weights//DeepSeek-V2-Lite-Chat"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--tasks', nargs='+', type=str, default=["mmlu"], help='usage: --tasks task1 task2')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)
