import torch
import torch.nn.functional as F
import sys
sys.path.append("../number_system")
from mx import finalize_mx_specs
from mx import mx_mapping
from mx.elemwise_ops import quantize_elemwise_op

def create_and_quantize_tensor(n, m, mean, std, mx_specs):
    # Create a tensor with standard normal distribution
    original_tensor = torch.randn(n, m)
    
    # Adjust the tensor to have the desired mean and std
    original_tensor = std * original_tensor + mean
    
    # Quantize the tensor to MX
    quantized_tensor = quantize_elemwise_op(
            original_tensor, mx_specs=mx_specs, round=mx_specs["round_output"]
        )
    dequantized_tensor =  quantized_tensor.float()

    # Calculate MSE
    mse_error = F.mse_loss(dequantized_tensor, original_tensor)
    
    return mse_error, original_tensor, quantized_tensor

# Example usage
if __name__ == "__main__":
    # Simple MX spec for MXFP6 weights+activations
    mx_specs = {
        'w_elem_format': 'fp4_e2m1',
        'a_elem_format': 'fp4_e2m1',
        'block_size': 32,
        'bfloat': 10,
        'custom_cuda': False,
        'quantize_backprop': False,
    }
    mx_specs = finalize_mx_specs(mx_specs)
    mse_error, original_tensor, quantized_tensor = create_and_quantize_tensor(128, 128, mean=10, std=1, mx_specs=mx_specs)
    print(f"MSE Error: {mse_error}")
