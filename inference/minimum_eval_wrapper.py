# streamlined from https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/models/huggingface.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

eval_logger = logging.getLogger(__name__)


@register_model("minimum")
class MinimumEvalWrapper(TemplateLM):
    """
    Minimum code to interface lm-eval for easy modification.    
    NOTE: hard-coded to batch_size=1 to avoid complex batching logic
    """
    def __init__(
        self,
        model,
        tokenizer,
        max_seq_length: Optional[int] = 4096,  # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L44
        device = "cuda",  # TODO: correctly set devices for TP>=2 cases
        softmax_dtype = None  # TODO: use float32 to get higher acc?
    ):
        super().__init__()
        self._max_seq_length = max_seq_length
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self.softmax_dtype = softmax_dtype

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def device(self):
        return self._device

    # NOTE: do not define self.world_size, otherwise evaluator will assume DP
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/evaluator.py#L540
    # here we hide TP inside model, invisble to lm-eval, similar to `VLLM`
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/models/vllm_causallms.py#L108

    def tok_encode(self, string: str, **kwargs):
        tokenizer = self._tokenizer
        tokens = tokenizer.encode(string)
        # some internal logic in lm-eval expects `encoded` to be a list, not tensor
        return tokens

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, x):
        """
        :param x: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        logits = self._model(x)  # just run prefill
        # print("logits.shape inside _model_call()", logits.shape)
        return logits

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        assert contlen and inplen, (
            "Must pass input len and cont. len to select scored logits for causal LM"
        )
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]
        return logits

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        sort_by_longest: bool = True,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # Simplified for batch_size=1: no Collator, just loop over requests
        # NOTE: can be 2x slower than HFLM even both with batch_size=1, due to missing the `logits_cache` feature
        # by `Collator` that saves computations for shared-prefix contexts. Here always recomputes for every query.
        # ref https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/models/huggingface.py#L1082

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        # Track original indices.
        indexed_requests = list(enumerate(requests))

        if sort_by_longest:
            # Sort requests by descending length to catch OOM early,
            # but recover the original order in the returned results.
            # Verified that output accuracy score is not affected.
            indexed_requests = sorted(indexed_requests, key=lambda r: len(r[1][1]), reverse=True)

        res = [None] * len(requests)  # Prepare a placeholder for results in original order
        for orig_idx, request in indexed_requests:
            request_str, context_enc, continuation_enc = request
            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            total_length = len(context_enc) + len(continuation_enc)
            if total_length > self.max_length + 1:
                eval_logger.warning(
                    f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                    f"exceeds model's maximum length ({self.max_length}). "
                    f"Truncating {total_length - self.max_length + 1} tokens from the left."
                )
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                dtype=torch.long,
                device=self.device,
            )
            inplen = inp.shape[0]
            cont_toks = continuation_enc
            contlen = len(cont_toks)  # for most tasks contlen==1, i.e. look at single next token

            inp = inp.unsqueeze(0)  # # [seq] -> [1, seq]
            logits = F.log_softmax(
                self._model_call(inp),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [1, inplen, vocab]
            logits = logits[0]  # [inplen, vocab]

            # For batch_size=1, padding_len_inp = inplen, so ctx_len = inplen
            ctx_len = inplen
            logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
            logits = logits.unsqueeze(0)  # [1, seq, vocab]

            greedy_tokens = logits.argmax(dim=-1)
            # print("logits.shape", logits.shape)
            # print("greedy_tokens.shape", greedy_tokens.shape)

            cont_toks_tensor = torch.tensor(
                cont_toks, dtype=torch.long, device=self.device
            ).unsqueeze(0)  # [1, seq]
            max_equal = (
                greedy_tokens[:, -cont_toks_tensor.shape[1]:] == cont_toks_tensor
            ).all()

            logits_gathered = torch.gather(
                logits, 2, cont_toks_tensor.unsqueeze(-1)
            ).squeeze(-1)  # [1, seq]

            answer = (float(logits_gathered.sum()), bool(max_equal))
            res[orig_idx] = answer

            if request_str is not None:
                self.cache_hook.add_partial(
                    "loglikelihood", request_str, answer
                )
            pbar.update(1)

        pbar.close()
        return res


    # # for PPL evals like `wikitext`
    # # taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/models/huggingface.py#L946
    # # Also see discussion for different perplexity calculation:
    # # https://github.com/EleutherAI/lm-evaluation-harness/issues/2170
    # # https://github.com/EleutherAI/lm-evaluation-harness/issues/1471
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self._tokenizer.eos_token_id,  # TODO: correct?
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls = []
        batch_size = 1  # WARNING: force bs=1 here
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial(
                "loglikelihood_rolling", (string,), request_total
            )

        return loglikelihoods


    # below are only needed for generative tasks, which can also be done by pure offline script
    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception('unimplemented')

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        raise Exception('unimplemented')

    @property
    def max_gen_toks(self):
        raise Exception('unimplemented')
