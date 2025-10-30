import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def fake_groupwise_token_asymmetric_quantization(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ).float()
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / (2**quantize_bit - 1)
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input


def fake_groupwise_channel_asymmetric_quantization_new(
    input: torch.Tensor, quantize_bit, group_size=128
):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype

    # group_size = 128
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    ).float()
    input = input.view(batch, seq_len, num_head * sep_dim)
    group_num = input.shape[1] // group_size

    fixed_input = input.view(batch,group_num, group_size, num_head * sep_dim)
    mx, mn = fixed_input.max(dim=-2)[0], fixed_input.min(dim=-2)[0]
    mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
    
    scale = (mx - mn) / (2**quantize_bit - 1)
    quantized_input = (fixed_input - mn) / scale
    quantized_input = F.relu(quantized_input)
    rounded_input = quantized_input.round_()
    dequantized_input = rounded_input * scale + mn
    dequantized_input = dequantized_input.view(batch,group_num * group_size,num_head, sep_dim)
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape

    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input


def fake_uniformquantization(input: torch.Tensor, quantize_bit):
    shape = input.shape
    dtype = input.dtype
    input = input.reshape(-1)
    input = input.float()  # convert to 32bits to avoid max - min = inf
    min, max = input.min(), input.max()
    step = (max - min) / (pow(2, quantize_bit) - 1)
    # print("before min max:",min,max,step)
    quantized_input = torch.round((input - min) / step)
    # print("after min max:",quantized_input.min(),quantized_input.max())
    # print("quantized isnan:",torch.any(torch.isnan(quantized_input)))
    dequantized_input = (quantized_input * step) + min
    returning_input = dequantized_input.reshape(shape)
    returning_input = returning_input.type(dtype)
    # print("isnan:",torch.any(torch.isnan(returning_input)))
    # while(True):
    #     pass
    return returning_input



def fake_dense_sparse_uniformquantization(input: torch.Tensor, quantize_bit, left):
    shape = input.shape
    dtype = input.dtype
    input = input.reshape(-1)
    input = input.float()  # convert to 32bits to avoid max - min = inf
    sortedtensor, indices = torch.sort(input)
    left_num = int(len(sortedtensor) * left / 2)
    sortedtensor[left_num:-left_num] = fake_uniformquantization(
        sortedtensor[left_num:-left_num], quantize_bit
    )
    input[indices] = sortedtensor
    input = input.reshape(shape)
    input = input.type(dtype)
    return input


def fake_groupwise_token_asymmetric_quantization_cluster(input,cluster_num,group_size=128):
    batch, num_head, seq_len, sep_dim = input.shape
    dtype = input.dtype
    input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    num_groups = (sep_dim * num_head) // group_size
    if num_groups * group_size != input.shape[-1]:
        raise ValueError("group_size should be a factor of the last dimension size")

    input_in_groups = input.view(batch, seq_len, num_groups, group_size)

    mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
    mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

    scale = (mx - mn) / cluster_num
    input_in_groups = (input_in_groups - mn) / scale
    input_in_groups = F.relu(input_in_groups)
    rounded_input_in_groups = input_in_groups.round_()
    dequantized_input_in_groups = rounded_input_in_groups * scale + mn
    dequantized_input = dequantized_input_in_groups.view(
        batch, seq_len, num_head, sep_dim
    )
    dequantized_input = dequantized_input.permute(0, 2, 1, 3)
    dequantized_input = dequantized_input.type(dtype)
    # reshape the input back to its original shape
    input = input.view(batch, seq_len, num_head, sep_dim)
    input = input.permute(0, 2, 1, 3).contiguous().type(dtype)
    return dequantized_input
def gearsl_channelQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    quantized_output = gearl_channelQ(input, quantize_bit, group_size,rank,loop)
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    input.scatter_(-1, smallest_indices, smallest_value)
    input.scatter_(-1, largest_indices, largest_value)
    

    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    input = input.half()
    quantized_output = quantized_output.half()

    
    return quantized_output

def gearslkivi_channelQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    sparsity_num = int(element_num * sparsity)
    # print(sparsity_num,sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    quantized_output = gearlkivi_channelQ(input, quantize_bit, group_size,rank,loop)
    input = input = (
        input.permute(0, 1, 3, 2).contiguous().view(batch, sep_dim * num_head, seq_len)
    )
    input.scatter_(-1, smallest_indices, smallest_value)
    input.scatter_(-1, largest_indices, largest_value)
    

    input = input.view(batch, num_head, sep_dim, seq_len).permute(0, 1, 3, 2)
    input = input.half()
    quantized_output = quantized_output.half()

    
    return quantized_output

def gearsl_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()

    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    input = input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3) 
    quantized_output = gearl_tokenQ(input, quantize_bit, group_size,rank,loop)
    # Restore the original values at the smallest and largest k indices
    quantized_output = quantized_output = (
        quantized_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    quantized_output.scatter_(-1, smallest_indices, smallest_value)
    quantized_output.scatter_(-1, largest_indices, largest_value)
    

    quantized_output = quantized_output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    quantized_output = quantized_output.half()
    return quantized_output 

def gearslkivi_tokenQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    cloned_input = input.clone()
    output = gears_tokenQ(input, quantize_bit, group_size,sparsity)

    error = cloned_input - output
    error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    return output + error_lr

def gearslkivi_channelQ_new(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()
    cloned_input = input.clone()
    output = gears_channelQ(input, quantize_bit, group_size,sparsity)

    error = cloned_input - output
    error_lr = fake_poweriteration_group(error, loop, rank, input.device, None, None)
    return output + error_lr

def gearslkivi_tokenQ(input, quantize_bit, group_size=128,sparsity=0.0,rank = 0,loop=1):
    input = input.float()

    batch, num_head, seq_len, sep_dim = input.shape
    element_num = batch * num_head * seq_len * sep_dim
    # input = input.reshape(-1)
    sparsity_num = int(element_num * sparsity)
    sparsity_pertoken = int(sparsity_num / batch / seq_len/2)
    input = input = (
        input.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )

    # Find the indices of the smallest k elements along the last dimension
    smallest_value, smallest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=False)
    # Find the indices of the largest k elements along the last dimension
    largest_value, largest_indices = torch.topk(input, sparsity_pertoken, dim=-1, largest=True)
    average = input.mean(dim=-1, keepdim=True)
    expanded_average = average.expand_as(input)
    index_helper = torch.arange(input.size(-1), device=input.device).expand_as(input)
    # Set the smallest k elements to the average value
    input.scatter_(-1, smallest_indices, expanded_average.gather(-1, smallest_indices))

    # Set the largest k elements to the average value
    input.scatter_(-1, largest_indices, expanded_average.gather(-1, largest_indices))
    input = input.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3) 
    quantized_output = gearlkivi_tokenQ(input, quantize_bit, group_size,rank,loop)
    # Restore the original values at the smallest and largest k indices
    quantized_output = quantized_output = (
        quantized_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, sep_dim * num_head)
    )
    quantized_output.scatter_(-1, smallest_indices, smallest_value)
    quantized_output.scatter_(-1, largest_indices, largest_value)
    

    quantized_output = quantized_output.view(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    quantized_output = quantized_output.half()
    return quantized_output


def tokenwise_gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = cloned_input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,

                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr

def gearlkivi_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    
    error = input - output
    #### TODO some changes here
    # error = error.permute(0, 1, 3, 2).contiguous().view(bsz, sep_dim * num_head, seq_len)
    # group_num = seq_len // group_size
    # error = error.view(bsz, sep_dim * num_head, group_num, group_size)
    
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
                                )
    # error_lr = error_lr.view(bsz, sep_dim, num_head, group_num*group_size).permute(0, 2, 3, 1).contiguous().view(bsz, num_head, group_num*group_size, sep_dim)
    
    return output + error_lr
def gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr
def tokenwise_gearlkivi_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    cloned_input = input.clone()
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = cloned_input - output
    # error = error.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, sep_dim * num_head)
    # num_groups = (sep_dim * num_head) // group_size
    # error = error.view(bsz, seq_len, num_groups, group_size)
    error_lr = fake_poweriteration_group(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None,
 
                                )
    # error_lr = error_lr.view(bsz, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output + error_lr

def gearl_channelQ(input, quantize_bit, group_size=128,r=0,loop=1):
    bsz, num_head, seq_len, sep_dim = input.shape
    output = fake_groupwise_channel_asymmetric_quantization_new(
        input, quantize_bit, group_size
    )
    error = input - output

    error_lr = fake_poweriteration_batchwise(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
    return output + error_lr

def gearl_tokenQ(input, quantize_bit, group_size=128,r=0,loop=1):
    output = fake_groupwise_token_asymmetric_quantization(
        input, quantize_bit, group_size
    )
    error = input - output
    error_lr = fake_poweriteration_batchwise(error,
                                loop,
                                r,
                                input.device,
                                None,
                                None
                                )
    return output + error_lr

def compress_insert_function(
    previous_key,
    previous_value,
    compress_config,
    layer_idx,
    pbase1=None,
    qbase1=None,
    pbase2=None,
    qbase2=None,
    prefill=None,
):
    batch, num_head, seq_len, sep_dim = previous_key.shape
    if compress_config.token_preserving[layer_idx] == True:
        starting_idx = int(compress_config.start_saving[layer_idx] * seq_len)
        locality_idx = int(compress_config.locality_saving[layer_idx] * seq_len)
    else:
        starting_idx = int(0)
        locality_idx = -seq_len
    # print("starting_idx:", starting_idx, "locality_idx:", locality_idx,compress_config.token_preserving[layer_idx],batch, num_head, seq_len, sep_dim)
    if compress_config.compress_method[layer_idx] == "channelQfixed":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            seq_len,
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                seq_len,
            )
    
    
    if compress_config.compress_method[layer_idx] == "tokenQfixed":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            int(num_head * sep_dim),
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                int(num_head * sep_dim),
            )


    if compress_config.compress_method[layer_idx] == "kcvtQfixed":
        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            seq_len,
        )
        if previous_value is not None:
            previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
                previous_value[:, :, starting_idx:-locality_idx, :],
                compress_config.quantize_bit[layer_idx],
                int(num_head * sep_dim),
            )


    if compress_config.compress_method[layer_idx] == "KIVI":


        previous_key[:, :, starting_idx:-locality_idx, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
        previous_value[:, :, starting_idx:-locality_idx, :] = fake_groupwise_token_asymmetric_quantization(
            previous_value[:, :, starting_idx:-locality_idx, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )

    if compress_config.compress_method[layer_idx] == "Flexgen":
        residual_length = seq_len % compress_config.group_size[layer_idx]

        previous_key[:, :, 0:-residual_length, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_key[:, :, 0:-residual_length, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
        previous_value[:, :, 0:-residual_length, :] = fake_groupwise_channel_asymmetric_quantization_new(
            previous_value[:, :, 0:-residual_length, :],
            compress_config.quantize_bit[layer_idx],
            compress_config.group_size[layer_idx]
        )
    return previous_key, previous_value


# iteratively compress the input tensor and simluate the error by low rank approximation
def fake_outquant_with_lrap_iter(tensor, quantize_bit, rank, loop, left, iter):
    lrap_error = tensor
    batch, num_head, seq_len, sep_dim = tensor.shape
    p_base = [torch.rand(sep_dim * num_head, rank).to(tensor.device)]
    q_base = [torch.rand(batch * seq_len, rank).to(tensor.device)]
    for i in range(iter):
        tensor_quantized = fake_dense_sparse_uniformquantization(
            lrap_error, quantize_bit, left
        )
        tensor_error = tensor - tensor_quantized
        tensor_error_lrap = fake_poweriteration(
            tensor_error, loop, rank, tensor_quantized.device, p_base, q_base
        )
        lrap_error = tensor - tensor_error_lrap
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


def fake_outquant_with_lrap(tensor, quantize_bit, rank, loop, left):
    tensor_quantized = fake_dense_sparse_uniformquantization(tensor, quantize_bit, left)
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


def fake_quant_with_lrap(tensor, quantize_bit, rank, loop):
    tensor_quantized = fake_uniformquantization(tensor, quantize_bit)
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


def fake_group_channel_quant_with_lrap(tensor, quantize_bit, rank, loop):
    tensor_quantized = fake_groupwise_channel_asymmetric_quantization(
        tensor, quantize_bit, 128
    )
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return


def fake_group_token_quant_with_lrap(tensor, quantize_bit, rank, loop):
    tensor_quantized = fake_groupwise_token_asymmetric_quantization(
        tensor, quantize_bit, 128
    )
    tensor_error = tensor - tensor_quantized
    tensor_error_lrap = fake_poweriteration(
        tensor_error, loop, rank, tensor_quantized.device, None, None
    )
    tensor_return = tensor_quantized + tensor_error_lrap
    return tensor_return



