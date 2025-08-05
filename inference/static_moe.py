from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class StaticMoEArgs:
    dim: int = 64
    intermediate_size: int = 342
    num_experts: int = 8
    num_activated_experts: int = 2


class ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim))
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.dim, config.intermediate_size))
        self.w3 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim))

    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        w1_weights = self.w1[expert_indices] # [T, A, D, D]
        w3_weights = self.w3[expert_indices] # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum('ti,taoi -> tao', x, w1_weights))
        x3 = torch.einsum('ti, taoi -> tao', x, w3_weights)
        expert_outs =  torch.einsum('tao, taio -> tai', (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = None  # take from dynamic MoE
        self.shared_experts = None # take from dynamic MoE
        self.cond_ffn = ConditionalFeedForward(config) # overwrite by dynamic MoE

        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)

        expert_weights, expert_indices = self.gate(x)

        expert_outs = self.cond_ffn(x, expert_indices)
        y = torch.einsum('tai,ta -> ti', expert_outs, expert_weights)
        z = self.shared_experts(x)
        # TODO: need allreduce for multi-device case
        return (y + z).view(shape)


def dynamic_moe_to_static(dynamic_args, moe_layer, device=None):
    static_config = StaticMoEArgs(
        num_experts=dynamic_args.n_routed_experts,
        num_activated_experts=dynamic_args.n_activated_experts,
        dim=dynamic_args.dim,
        intermediate_size=dynamic_args.moe_inter_dim
    )

    with torch.device(device):
        static_moe_layer = MOEFeedForward(static_config)
    static_moe_layer.gate = moe_layer.gate

    # merge expert list to 3D weight matrix
    for i in range(len(moe_layer.experts)):
        assert moe_layer.experts[i].w1.bias is None
        assert moe_layer.experts[i].w2.bias is None
        assert moe_layer.experts[i].w3.bias is None
        static_moe_layer.cond_ffn.w1.data[i] = moe_layer.experts[i].w1.weight.data
        static_moe_layer.cond_ffn.w2.data[i] = moe_layer.experts[i].w2.weight.data
        static_moe_layer.cond_ffn.w3.data[i] = moe_layer.experts[i].w3.weight.data

    static_moe_layer.shared_experts = moe_layer.shared_experts

    return static_moe_layer


@torch.inference_mode()
def test_moe_replacement():
    from model import Linear, MoE, Gate

    @dataclass
    class DynamicModelArgs:
        dim: int = 64
        moe_inter_dim: int = 44
        n_routed_experts: int = 16
        n_shared_experts: int = 1
        n_activated_experts: int = 8
        n_expert_groups: int = 8
        n_limited_groups: int = 4
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_scale: float = 1.0

    args = DynamicModelArgs()

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    device = "cuda:0"
    with torch.device(device):
        moe_layer = MoE(args)

    def init_weights_normal(m):
        if isinstance(m, (Linear, Gate)):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)

    moe_layer.apply(init_weights_normal)

    batch_size = 3
    x = torch.randn(batch_size, args.dim, device=device, dtype=dtype)
    y = moe_layer(x)

    static_moe_layer = dynamic_moe_to_static(args, moe_layer, device=device)
    y_static = static_moe_layer(x)

    absdiff = torch.abs(y_static - y).to(torch.float32).cpu()
    print("mean & max abs error: ", float(absdiff.mean()), float(absdiff.max()))
    torch.testing.assert_close(y_static, y, rtol=1e-2, atol=1e-3)

if __name__ == "__main__":
    test_moe_replacement()
