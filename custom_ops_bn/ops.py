import torch
from torch import Tensor
from typing import Optional, Literal, Tuple

__all__ = ["batchnorm", "BatchNormCUDAFunction", "BatchNormTritonFunction"]


# register as a custom operator
@torch.library.custom_op("custom_ops_bn::triton_batchnorm_forward", mutates_args=("running_mean", "running_var"))
def triton_batchnorm_forward(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    from .triton_batchnorm import (
        batchnorm_forward_training_kernel,
        batchnorm_forward_inference_kernel
    )

    N, C = input.shape[0], input.shape[1]
    spatial_dim = input.numel() // (N * C)

    output = torch.empty_like(input)
    save_mean = torch.empty(C, dtype=input.dtype, device=input.device)
    save_invstd = torch.empty(C, dtype=input.dtype, device=input.device)

    # Get strides
    stride_n = input.stride(0)
    stride_c = input.stride(1)
    stride_s = 1 if input.dim() == 4 else 0

    BLOCK_SIZE = 1024
    grid = (C,)

    if training:
        batchnorm_forward_training_kernel[grid](
            input, gamma, beta, output, save_mean, save_invstd,
            running_mean, running_var,
            N, C, spatial_dim,
            momentum, eps,
            stride_n, stride_c, stride_s,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        save_mean.copy_(running_mean)
        batchnorm_forward_inference_kernel[grid](
            input, gamma, beta, running_mean, running_var,
            output, save_invstd,
            N, C, spatial_dim,
            eps,
            stride_n, stride_c, stride_s,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output, save_mean, save_invstd


@torch.library.custom_op("custom_ops_bn::triton_batchnorm_backward", mutates_args=())
def triton_batchnorm_backward(
    grad_output: Tensor,
    input: Tensor,
    gamma: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    from .triton_batchnorm import batchnorm_backward_kernel

    # Ensure inputs are contiguous
    grad_output = grad_output.contiguous()
    input = input.contiguous()

    N, C = input.shape[0], input.shape[1]
    spatial_dim = input.numel() // (N * C)

    grad_input = torch.empty_like(input)
    grad_gamma = torch.zeros(C, dtype=gamma.dtype, device=gamma.device)
    grad_beta = torch.zeros(C, dtype=gamma.dtype, device=gamma.device)

    # Get strides
    stride_n = input.stride(0)
    stride_c = input.stride(1)
    stride_s = 1 if input.dim() == 4 else 0

    BLOCK_SIZE = 1024
    grid = (C,)

    batchnorm_backward_kernel[grid](
        grad_output, input, gamma, save_mean, save_invstd,
        grad_input, grad_gamma, grad_beta,
        N, C, spatial_dim,
        stride_n, stride_c, stride_s,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_input, grad_gamma, grad_beta

# connect custom operator with torch.autograd.Function
class BatchNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-5):
        output, save_mean, save_invstd = torch.ops.custom_ops_bn.triton_batchnorm_forward(
            input, gamma, beta, running_mean, running_var,
            training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.custom_ops_bn.triton_batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None


class BatchNormCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-5):
        output, save_mean, save_invstd = torch.ops.custom_ops_bn.batchnorm_forward(
            input, gamma, beta, running_mean, running_var,
            training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.custom_ops_bn.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None

def batchnorm(
    input: Tensor,
    gamma: Optional[Tensor] = None,
    beta: Optional[Tensor] = None,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
    backend: Literal['cuda', 'triton'] = 'cuda'
) -> Tensor:
    
    C = input.size(1)

    if gamma is None:
        gamma = torch.ones(C, dtype=input.dtype, device=input.device)
    if beta is None:
        beta = torch.zeros(C, dtype=input.dtype, device=input.device)
    if running_mean is None:
        running_mean = torch.zeros(C, dtype=input.dtype, device=input.device)
    if running_var is None:
        running_var = torch.ones(C, dtype=input.dtype, device=input.device)

    # Select backend
    if backend == 'cuda':
        return BatchNormCUDAFunction.apply(
            input, gamma, beta, running_mean, running_var,
            training, momentum, eps
        )
    elif backend == 'triton':
        return BatchNormTritonFunction.apply(
            input, gamma, beta, running_mean, running_var,
            training, momentum, eps
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'cuda' or 'triton'.")

