Modified from https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py

- Fix minor bugs to be compatible with latest huggingface version (see NOTE(jzhuang) comments in ./modeling_deepseek.py file)
- Keep backend-agnostic, only default torch/aten ops, not using custom GPU/NPU kernels
