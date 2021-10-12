import math
import time
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import swextension
from typing import List
from torch import Tensor


from torch.optim import Optimizer

def swadam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient

        ###################  sw  ########################################
        swextension.swmul(exp_avg,beta1)
        swextension.swadd(exp_avg, grad, 1 - beta1)
        
        swextension.swmul(exp_avg_sq,beta2)
        swextension.swaddcmul(exp_avg_sq, grad, grad, 1 - beta2)
        #################################################################

        ###################  orgin  ######################################
        # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        ###################################################################

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class swAdam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(swAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(swAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            swadam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps']
                   )
        return loss



# class swAdam(Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=1e-2, amsgrad=False):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay, amsgrad=amsgrad)
#         super(swAdam, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(swAdam, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
                
#                 # Perform stepweight decay
#                 # p.mul_(1 - group['lr'] * group['weight_decay'])
#                 swextension.swmul(p,1 - group['lr'] * group['weight_decay'])
                

#                 # Perform optimization step
#                 grad = p.grad
#                 if grad.is_sparse:
#                     raise RuntimeError('swAdam does not support sparse gradients')
#                 amsgrad = group['amsgrad']

#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                     if amsgrad:
#                         # Maintains max of all exp. moving avg. of sq. grad. values
#                         state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 if amsgrad:
#                     max_exp_avg_sq = state['max_exp_avg_sq']
#                 beta1, beta2 = group['betas']

#                 state['step'] += 1
#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']

#                 # Decay the first and second moment running average coefficient

#                 swextension.swmul(exp_avg,beta1)
#                 exp_avg.add_(grad, alpha=1 - beta1)
#                 # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

#                 swextension.swmul(exp_avg_sq,beta2)
#                 exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)
#                 # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)



#                 if amsgrad:
#                     # Maintains the maximum of all 2nd moment running avg. till now
#                     torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
#                     # Use the max. for normalizing running avg. of gradient
#                     denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
#                 else:
#                     denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

#                 step_size = group['lr'] / bias_correction1

#                 p.addcdiv_(exp_avg, denom, value=-step_size)

#         return loss


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
        return swLinearFunction.apply(input, self.weight, self.bias)


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
