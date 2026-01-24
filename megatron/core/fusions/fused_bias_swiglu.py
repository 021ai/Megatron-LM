# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


@jit_fuser
def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    """Computes the gradient for the SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed using the
            chain rule and the derivative of the SiLU activation function.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    """Computes the gradient for the biased SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasSwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            bias (torch.Tensor): Bias tensor to be added to input before SwiGLU.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying bias addition followed by SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_swiglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - Gradient with respect to the bias tensor
                - None for fp8_input_store parameter
        """
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_swiglu_back(grad_output, input, bias)
        return tmp, tmp, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return swiglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - None for fp8_input_store parameter
        """
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = swiglu_back(grad_output, input)
        return tmp, None, None

from transformer_engine.pytorch.cpu_offload import set_offloading_param, get_fine_grained_offload_handler, has_acivation_offloading_param

class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store, fine_grained_offload: bool = False):
        fine_grained_offload_handler = get_fine_grained_offload_handler()
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if fine_grained_offload and not has_acivation_offloading_param(input_for_backward) and not fine_grained_offload_handler.is_last_2_pipeline_parallel_stage() and not fine_grained_offload_handler.is_last_batch_last_layer():
            set_offloading_param(input_for_backward, 'fine_grained_offloading', 'moe_fused_swiglu_input')
            ctx.tensor_tag = fine_grained_offload_handler.register_offload(input_for_backward)
            ctx.save_for_backward(weights)
            ctx.input_for_backward = input_for_backward    
        else:
            if fine_grained_offload and has_acivation_offloading_param(input_for_backward):
                # [fine_grained_offload mode] in recomputing fwd phase (2nd fwd phase) 
                tensor_tag = fine_grained_offload_handler.get_tag_from_name('moe_fused_swiglu_input')                
                input_for_backward = fine_grained_offload_handler.get_reloaded(tensor_tag)
                input = input_for_backward.to(input.dtype) if fp8_input_store else input_for_backward
            
            ctx.tensor_tag = None
            ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.fine_grained_offload = fine_grained_offload
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        fine_grained_offload_handler = get_fine_grained_offload_handler()
        if ctx.tensor_tag != None:
            (weights, ) = ctx.saved_tensors
            input = ctx.input_for_backward
            assert not fine_grained_offload_handler.is_last_2_pipeline_parallel_stage() and not fine_grained_offload_handler.is_last_batch_last_layer()
            input = fine_grained_offload_handler.get_reloaded(ctx.tensor_tag)
        else:
            input, weights = ctx.saved_tensors
        
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None, None


def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased SwiGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SwiGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SwiGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SwiGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SwiGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False, fine_grained_offload : bool = False):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input_view = input.view(-1, ori_shape[-1])
    from megatron.core.utils import make_viewless_tensor
    input_view = make_viewless_tensor(inp=input_view, requires_grad=input_view.requires_grad, keep_graph=True)
    if hasattr(input, "fine_grained_offloading"):
        setattr(input_view, "fine_grained_offloading", input.fine_grained_offloading)
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        output = WeightedSwiGLUFunction.apply(input_view, weights, fp8_input_store, fine_grained_offload)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
