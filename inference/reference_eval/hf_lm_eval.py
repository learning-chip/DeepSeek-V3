"""
Dependency
    pip install transformers==4.55.0

Usage

    python hf_lm_eval.py --tasks wikitext | tee dsv2_hfref_eval_wikitext.log
    python hf_lm_eval.py --tasks mmlu | tee dsv2_hfref_eval_mmlu.log
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

def main(args):
    print("args: ", args)
    model_path = args.model_path

    torch.cuda.set_device(args.device)

    model = DeepseekV2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = HFLM(
        pretrained=model,
        tokenizer=tokenizer
    )

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
