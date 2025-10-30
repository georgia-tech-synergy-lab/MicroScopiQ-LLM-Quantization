import torch
import torch.nn as nn
import numpy as np
import copy

import sys
sys.path.append("../number_system")
from mx import MXLinear, LayerNorm
from mx import Conv1d, Conv2d
from mx import gelu, simd_split, simd_add
from mx import add_mx_args, get_mx_specs

from transformers import pytorch_utils

def quantize_model(model, mx_specs):
    """
    Recursively quantize a pretrained single-precision model
    """
    # Quantize Conv2d
    if type(model) == nn.Conv2d:
        quant_mod = Conv2d(model.in_channels, model.out_channels,
                            model.kernel_size, model.stride,
                            model.padding, model.dilation, 
                            model.groups, model.bias,
                            mx_specs)
        
        quant_mod.weight.data = model.weight.data.clone()
        if model.bias is not None:
            quant_mod.bias.data = model.bias.data.clone()
        
        return quant_mod
    
    # Quantize Linear
    elif type(model) == nn.Linear:
        print(model, "running")
        quant_mod = MXLinear(model.in_features, model.out_features,
                            True, mx_specs)

        quant_mod.weight.data = model.weight.data.clone()
        if model.bias is not None:
            quant_mod.bias.data = model.bias.data.clone()

        return quant_mod

    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, mx_specs))
        return nn.Sequential(*mods)

    elif type(model) == nn.ModuleList:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, mx_specs))
        return nn.Sequential(*mods)

    elif isinstance(model, nn.Sequential):
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, mx_specs))
        return nn.Sequential(*mods)

    else:
        # recursively use the quantized module to replace the single-precision module
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and attr != 'base_model' and attr!= 'lm_head':
                setattr(q_model, attr, quantize_model(mod, mx_specs))

        return q_model

