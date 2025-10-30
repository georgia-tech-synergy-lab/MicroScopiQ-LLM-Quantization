import numpy as np
import torch
import torch.nn as nn
import random
import sys
sys.path.append("../number_system/")
from mx.specs import mx_assert_test
from mx.formats import (
        RoundingMode,
        ElemFormat,
        FP32_EXPONENT_BIAS,
        FP32_MIN_NORMAL,
        _get_format_params
)
from mx.elemwise_ops import (
        _safe_lshift, _safe_rshift,
        _round_mantissa,
        _quantize_elemwise_core
)

from mx.specs import finalize_mx_specs

def quantize_mx_outlier_hessian(
    A,
    inlier_scale_bits,
    outlier_scale_bits,
    inlier_elem_format,    # can be None for no quantization
    outlier_elem_format,    # can be None for no quantization
    shared_exp_method="max",
    std_dev = 2,
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    prune_inliers = False,
    custom_cuda=False
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
    
    # Estimate axis to calculate shared exponent
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Extract Outliers position for each block
    outlier_pos = _extract_outlier_indices(A, std_dev, shared_exp_axes)
    num_outliers = (outlier_pos[::block_size, :, :].sum(axis=-2, keepdim=False).flatten()).to(torch.int8)
    # print(num_outliers.shape)
    #print("Outlier_Pos", outlier_pos)
    # Get inliers based on complement on outlier position
    inlier_val = A * (1.0 - outlier_pos)
    outlier_val = A * outlier_pos

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
    shared_exp_in[shared_exp_in < -scale_emax_in] = -20 if (-scale_emax_in < -20) else -scale_emax_in

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
    shared_exp_out[shared_exp_out < -scale_emax_out] = -20 if (-scale_emax_out < -20) else -scale_emax_out
    
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
    # print(A.size())
    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A, num_outliers
def quantize_mx_outlier_v1(
    A,
    inlier_scale_bits,
    outlier_scale_bits,
    inlier_elem_format,    # can be None for no quantization
    outlier_elem_format,    # can be None for no quantization
    shared_exp_method="max",
    std_dev = 2,
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False
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
    
    # Estimate axis to calculate shared exponent
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Extract Outliers position for each block
    outlier_pos = _extract_outlier_indices(A, std_dev, shared_exp_axes)
    
    #print("Outlier_Pos", outlier_pos)
    # Get inliers based on complement on outlier position
    inlier_val = A * (1.0 - outlier_pos)
    outlier_val = A * outlier_pos

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
    shared_exp_in[shared_exp_in < -scale_emax_in] = -20 if (-scale_emax_in < -20) else -scale_emax_in

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
    shared_exp_out[shared_exp_out < -scale_emax_out] = -20 if (-scale_emax_out < -20) else -scale_emax_out
    
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

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class MXQuantizer(nn.Module):

    def __init__(self, shape=1):
        super(MXQuantizer, self).__init__()
        self.mx_specs =  {
        'w_elem_format': 'int2',
        'a_elem_format': 'fp16',
        'block_size': 128,
        'custom_cuda': False,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
        }

        self.mx_specs = finalize_mx_specs(self.mx_specs)

    def configure(
        self,
        inlier_scale_bits, outlier_scale_bits,
        inlier_elem_format, outlier_elem_format,    # can be None for no quantization
        shared_exp_method="max", std_dev = 2,
        axes=None, block_size=0,
        round="nearest", flush_fp32_subnorms=False,
        custom_cuda = False
    ):
        self.inlier_scale_bits = inlier_scale_bits
        self.inlier_elem_format = inlier_elem_format
        self.outlier_scale_bits = outlier_scale_bits
        self.outlier_elem_format = outlier_elem_format
        self.shared_exp_method = shared_exp_method
        self.std_dev = std_dev
        self.axes = axes
        self.block_size = block_size
        self.round = round
        self.flush_fp32_subnorms = flush_fp32_subnorms
        self.custom_cuda = custom_cuda

    def find_params(self, x, weight=False):
        pass

    def quantize(self, x):
        if self.ready():
            return quantize_mx_outlier_v1(
                x,
                self.inlier_scale_bits,
                self.outlier_scale_bits,
                self.inlier_elem_format,    # can be None for no quantization
                self.outlier_elem_format,    # can be None for no quantization
                self.shared_exp_method,
                self.std_dev,
                self.axes,
                self.block_size,
                self.round,
                self.flush_fp32_subnorms,
                self.custom_cuda
            )
        return x

    def enabled(self):
        pass

    def ready(self):
        return True


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
    # print("**************************************")
    if axes is not None:
        # Mean and std along specific axes
        for axis in axes:
            mean = torch.mean(torch.abs(A), dim=axes, keepdim=True)
            std = torch.std(torch.abs(A), dim=axes, keepdim=True, unbiased=False)
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


