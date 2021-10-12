import math
import time
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
import numpy as np
import swextension


class swLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        # return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        d_grad = grad_output.clone()
        d_grad = swextension.swrelu_backward(input,d_grad)
        return d_grad

class swLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5, elementwise_affine = True):
        super(swLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return swLayerNormFunction.apply(input, self.normalized_shape, self.weight, self.bias, self.eps)
        # return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

class swReluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = swextension.swrelu_forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        d_grad = grad_output.clone()
        d_grad = swextension.swrelu_backward(input,d_grad)
        return d_grad
         
        
class swRelu(nn.Module):
    def __init__(self, inplace: bool = False):
        super(swRelu, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return swReluFunction.apply(input)

class swLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        outputs = swextension.swlinear_forward(input, weight)
        if bias is not None:
            # outputs[0] += bias.unsqueeze(0).expand_as(outputs[0])
            outputs[0] += bias

        variables = [input] + [weight] + [bias]
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        d_input, d_weight, d_bias = swextension.swlinear_backward(
            grad_output, *ctx.saved_variables
        )
        return d_input, d_weight, d_bias


class swLinear(nn.Module):
    def __init__(self, input_features, output_features, bais=False):
        super(swLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bais:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        st=time.perf_counter()
        x=swLinearFunction.apply(input, self.weight, self.bias)
        ed=time.perf_counter()
        # print(f'swLinearFunction timing: {ed - st}')
        return x


class swMHAFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, nheads, scaling):
        outputs = swextension.swmha_forward(input, weight, nheads, scaling)
        variables = [input] + [weight] + outputs[1:]
        ctx.constant = scaling
        ctx.save_for_backward(*variables)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        d_input, d_weight = swextension.swmha_backward(
            grad_output, *ctx.saved_variables, ctx.constant
        )
        return d_input, d_weight, None, None


class swMHA(nn.Module):
    def __init__(self, input_features, output_features, heads, scaling):
        super(swMHA, self).__init__()
        self.input_features = input_features
        self.heads = heads
        self.scaling = scaling
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return swMHAFunction.apply(input, self.weight, self.heads, self.scaling)
