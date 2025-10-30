"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Name:    mx_ops.py

Pytorch methods for MX quantization.

Usage Notes:
 - Use the "Exposed Methods" below to implement autograd functions
 - Use autograd functions to then implement torch.nn.Module(s)
 - Do *not* use methods in this file in Modules, they have no defined
   backwards pass and will block gradient computation.
 - Avoid importing internal function if at all possible.

Exposed Methods:
    quantize_mx_op - quantizes a tensor to MX format.

Internal Methods:
    _safe_lshift, _safe_rshift - fp16 compatible shifts
    _shared_exponents - Returns MX shared exponent for the passed tensor
    _reshape_to_blocks - tiles a tensor by splitting one dim into two
    _undo_reshape_to_blocks - undos the above reshaping
    _quantize_mx - quantizes a tensor to MX format
"""

import os
import torch
import numpy as np
import sys
from .specs import mx_assert_test
from .formats import (
        RoundingMode,
        ElemFormat,
        FP32_EXPONENT_BIAS,
        FP32_MIN_NORMAL,
        _get_format_params
)
from .elemwise_ops import (
        _safe_lshift, _safe_rshift,
        _round_mantissa,
        _quantize_elemwise_core
)


# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
def _extract_outlier_indices(A, std_dev = 1, axes=None):
    """
    Identifies outliers in the tensor A which are more than std_dev times
    the standard deviation away from the mean along the specified axes.
    
    Args:
    A (torch.Tensor): Input tensor.
    std_dev (float): Number of standard deviations to use as the threshold for detecting outliers.
    axes (int or tuple of ints, optional): Axis or axes along which to calculate mean and standard deviation. If None, the outliers are computed over the entire tensor.
    
    Returns:
    torch.Tensor: A tensor of the same shape as A, where 1 indicates an outlier and 0 indicates a non-outlier.
    """
    if axes is not None:
        # Mean and std along specific axes
        mean = torch.mean(A, dim=axes, keepdim=True)
        std = torch.std(A, dim=axes, keepdim=True)
        assert not torch.isnan(std).any(), "std contains NaN values"

    else:
        # Mean and std over the entire tensor
        mean = torch.mean(A)
        std = torch.std(A)
        # Expand mean and std to match the shape of A for broadcasting
        mean = mean.expand_as(A)
        std = std.expand_as(A)
    
    # Calculate the threshold boundaries for outliers
    lower_bound = mean - (std_dev * std)
    upper_bound = mean + (std_dev * std)
    # Identify outliers
    outliers = (A < lower_bound) | (A > upper_bound)
    
    # Convert boolean tensor to the same dtype as A (usually float or int)
    return outliers.type(A.dtype)


def _shared_exponents(A, method="max", axes=None, ebits=0):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        #shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_mx_outlier_v1(
    A,
    inlier_scale_bits,
    outlier_scale_bits,
    inlier_elem_format,    # can be None for no quantization
    outlier_elem_format,    # can be None for no quantization
    shared_exp_method="max",
    std_dev = 5,
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False,
):

    """Function used for MX* Outlier quantization
    """
    # Shortcut for no quantization
    if inlier_elem_format == None:
        return A

    assert(inlier_scale_bits > 0 and outlier_scale_bits > 0)
    
    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Get parameters of the inlier and outlier formats
    ebits_in, mbits_in, emax_in, max_norm_in, _ = _get_format_params(inlier_elem_format)
    ebits_out, mbits_out, emax_out, max_norm_out, _ = _get_format_params(outlier_elem_format)

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    # Extract Outliers position for each block
    outlier_pos = _extract_outlier_indices(A, std_dev, axes)
    #print("Outlier_Pos", outlier_pos)
    # Get inliers based on complement on outlier position
    inlier_val = A * (1.0 - outlier_pos)
    outlier_val = A * outlier_pos

    # Estimate axis to calculate shared exponent
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents for inliers
    shared_exp_in = _shared_exponents(
        inlier_val, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
    )

    # Flush subnormals to 0
    if flush_fp32_subnorms:
        inlier_val = inlier_val * (shared_exp_in > -FP32_EXPONENT_BIAS).type(inlier_val.dtype)

    # Offset the max exponent by the largest representable exponent
    # in the element data format

    shared_exp_in = shared_exp_in - emax_in
    scale_emax_in = 2**(inlier_scale_bits-1) - 1

    shared_exp_in[shared_exp_in > scale_emax_in] = float("NaN")
    shared_exp_in[shared_exp_in < -scale_emax_in] = -scale_emax_in

    # Scale inlier values
    inlier_val = inlier_val / (2**shared_exp_in)
    # Level-1 scaling of outliers
    outlier_val = outlier_val * (2**shared_exp_in)
    # Quantize inliers
    inlier_val = _quantize_elemwise_core(
                inlier_val, mbits_in, ebits_in, max_norm_in, round=round,
                allow_denorm=True, saturate_normals=True,
                custom_cuda=custom_cuda)

    # Dequantize inliers
    inlier_val = inlier_val * (2**shared_exp_in)
    assert not torch.isnan(inlier_val).any(), "inlier_val contains NaN values"
    assert not torch.isnan(outlier_val).any(), "outlier_val 1 contains NaN values"
    #*****************************************************
    # Get shared exponents for outliers
    shared_exp_out = _shared_exponents(
        outlier_val, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
    )
    assert not torch.isnan(shared_exp_out).any(), "shared_exp_out contains NaN values"
    # No need to check for subnorm for outliers, if they were subnormal, they wouldn't be outliers
    # if flush_fp32_subnorms:
    #     outlier_val = outlier_val * (shared_exp_out > -FP32_EXPONENT_BIAS).type(outlier_val.dtype)
    
    shared_exp_out = shared_exp_out - emax_out

    scale_emax_out = 2**(outlier_scale_bits-1) - 1
    # print("Scale EMAX OUt", scale_emax_out)
    shared_exp_out[shared_exp_out > scale_emax_out] = float("NaN")
    shared_exp_out[shared_exp_out < -scale_emax_out] = -scale_emax_out
    
    assert not torch.isnan(shared_exp_out).any(), "shared_exp_out contains NaN values"
    # Level-2 scaling of outliers
    # print(outlier_val[0,51,22,0], shared_exp_out[0,51,22,0])
    outlier_val = outlier_val / (2**shared_exp_out)
    # print(outlier_val[0,51,22,0])
    # print(list(zip(*torch.where(torch.isnan(outlier_val)))))
    assert not torch.isnan(outlier_val).any(), "outlier_val contains NaN values"
    # Quantize outliers
    outlier_val = _quantize_elemwise_core(
                outlier_val, mbits_out, ebits_out, max_norm_out, round=round,
                allow_denorm=True, saturate_normals=True,
                custom_cuda=custom_cuda)
    
    # Dequantize outliers using level-1 and level-2 scale factors
    outlier_val = (outlier_val * (2**shared_exp_out))/ (2**shared_exp_in)
    #*****************************************************

    # Reconstruct A
    A = inlier_val + outlier_val

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A

def _quantize_mx(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False,
):
    """Function used for MX* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Use quantize_mx_by_tile when there is only a single shared axis and
    # - The block size is small, OR
    # - The shared axis is not the innermost
    if A.device.type == "cuda" and custom_cuda and len(axes) == 1:
        axis = axes[0]
        if block_size == 0:
            block_size = A.shape[axis]

        if axis != len(A.shape) - 1 or block_size <= 32:
            A = A.contiguous()

            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_by_tile_func_cuda(
                A,
                scale_bits,
                ebits,
                mbits,
                max_norm,
                block_size,
                axis,
                flush_fp32_subnorms,
                RoundingMode[round],
            )
            return A


    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    if custom_cuda:
        # Custom CUDA code only supports a single axis
        if shared_exp_axes is None:
            axis = 0
        else:
            assert len(shared_exp_axes) == 1
            axis = shared_exp_axes[0]

        assert(shared_exp_method == "max")
        max_values = A.abs().max(dim=axis, keepdim=True).values

        A = A.contiguous()

        if A.device.type == "cuda":
            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_func_cuda(
                A, scale_bits, ebits, mbits, max_norm,
                max_values, axis,
                flush_fp32_subnorms, RoundingMode[round]);

        elif A.device.type == "cpu":
            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_func_cpp(
                A, scale_bits, ebits, mbits, max_norm,
                max_values, axis,
                flush_fp32_subnorms, RoundingMode[round]);

        else:
            raise ValueError("Unrecognized device type %s" % A.device.type)
    else:
        # Get shared exponents
        shared_exp = _shared_exponents(
            A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
        )

        # Flush subnormal FP32 inputs to zero
        if flush_fp32_subnorms:
            A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - emax

        scale_emax = 2**(scale_bits-1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        A = A / ((2**shared_exp) + 1e-6)

        A = _quantize_elemwise_core(
                A, mbits, ebits, max_norm, round=round,
                allow_denorm=True, saturate_normals=True,
                custom_cuda=custom_cuda)

        A = A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


def quantize_mx_op(
    A,
    mx_specs: dict,
    elem_format=None,
    block_size=None,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):
    mx_assert_test(mx_specs)

    if elem_format == None:
        return A
    elif type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)

    if block_size == None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]

    return _quantize_mx(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            custom_cuda=mx_specs["custom_cuda"])

def quantize_mx_outlier_op(
    A,
    mx_specs: dict,
    inlier_elem_format=None,
    outlier_elem_format=None,
    block_size=None,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):
    mx_assert_test(mx_specs)

    # Inlier Elements
    if inlier_elem_format == None:
        return A
    elif type(inlier_elem_format) is str:
        inlier_elem_format = ElemFormat.from_str(inlier_elem_format)
    
    # Outlier Elements
    if outlier_elem_format == None:
        return A
    elif type(outlier_elem_format) is str:
        outlier_elem_format = ElemFormat.from_str(outlier_elem_format)

    if block_size == None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        inlier_scale_bits = 4
    else:
        inlier_scale_bits = mx_specs["scale_bits"]

    outlier_scale_bits = inlier_scale_bits

    return _quantize_mx_outlier_v1(
            A, inlier_scale_bits,
            outlier_scale_bits, inlier_elem_format, 
            outlier_elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            custom_cuda=mx_specs["custom_cuda"])


if __name__ == "__main__":
    # mx_specs = {
    #     'w_elem_format': 'fp4_e2m1',
    #     'a_elem_format': 'fp4_e2m1',
    #     'block_size': 1,
    #     'bfloat': 10,
    #     'custom_cuda': False,
    #     'quantize_backprop': False,
    # }

    # mx_specs = finalize_mx_specs(mx_specs)

    torch.manual_seed(42)
    x = torch.randn(1, 2)
    print(_quantize_mx(x, 8, 'fp4', block_size=1, axes=[0], round=round, ))