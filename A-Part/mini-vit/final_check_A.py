from transformer import Transformer as TransformerActual
from transformerExpected import Transformer as TransformerExpected
from custom_names import param_names_dic_act

import copy
import torch
import os
import numpy as np

"""
implement your model in transformer.py
write your own param names in custom_names.py dict according to model structure.
do not modify check.py and transformerExpected.py
"""
# expected names dict
param_names_dic_exp = {
    "to_qkv_weight": "layers[0][0].fn.to_qkv.weight",
    "to_out_weight": "layers[0][0].fn.to_out[0].weight",
    "to_out_bias": "layers[0][0].fn.to_out[0].bias",
    "net0_weight": "layers[0][1].fn.net[0].weight",
    "net0_bias": "layers[0][1].fn.net[0].bias",
    "net2_weight": "layers[0][1].fn.net[2].weight",
    "net2_bias": "layers[0][1].fn.net[2].bias",
}


def check_eq(a, b, msg):
    if a.size() != b.size():
        print(f"{msg} check\033[1;31m failed!\033[0m size mismatch!")
        return 1
    # absolute err < 1e-4
    flag = torch.allclose(a, b, atol=1e-4, rtol=1.3e-4)
    if not flag:
        print(f"{msg} check\033[1;31m failed!\033[0m")
    else:
        print(f"{msg} check\033[1;32m passed!\033[0m")


def check_transformer(seed, batch, seq, dim, depth, heads, dim_head, mlp_dim):
    print("\033[1;35m checking transformer \033[0m")
    torch.manual_seed(seed)
    # init ref model
    transformer_expected = TransformerExpected(
        dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
    )
    # generate random input
    x = torch.rand(batch, seq, dim, dtype=torch.float, requires_grad=True)
    # ref model forward and backward
    out_expected = transformer_expected(x)
    grad = torch.rand(size=out_expected.shape, dtype=torch.float)
    out_expected.backward(grad)
    x_grad_expected = copy.deepcopy(x.grad)
    x.grad.zero_()
    # init custom model
    transformer_actual = TransformerActual(
        dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
    )

    for param_name in param_names_dic_exp.keys():
        eval(
            "transformer_actual." + param_names_dic_act[param_name]
        ).data = copy.deepcopy(
            eval("transformer_expected." + param_names_dic_exp[param_name]).data
        )
    # custom model forward and backward
    out_actual = transformer_actual(x)
    out_actual.backward(grad)
    x_grad_actual = x.grad
    # check value
    check_eq(x_grad_actual, x_grad_expected, "x_grad")
    check_eq(out_actual, out_expected, "y")
    for param_name in param_names_dic_exp.keys():
        check_eq(
            eval("transformer_actual." + param_names_dic_act[param_name]).grad,
            eval("transformer_expected." + param_names_dic_exp[param_name]).grad,
            param_name + "_grad",
        )


if __name__ == "__main__":
    seed = 233
    config_para = [
                    [96,     0.9,     128,    2,     4,     256,],
                    [32,     0.9,     768,    2,     6,    1024,],
                    [48,     0.9,    1152,    2,     5,    1792,],
                    [256,    0.9,      32,    2,     4,      64,],
                    [1024,   0.9,      64,    2,     2,      64,],
                  ]
    for config in config_para:
        _, _, dim, _, heads, mlp_dim = config
        batch = 6
        seq = 50
        dim_head = 64
        depth = 1
        print(config)
        check_transformer(seed, batch, seq, dim, depth, heads, dim_head, mlp_dim)
