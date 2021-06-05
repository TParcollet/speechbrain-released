import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.grad import conv2d_input, conv2d_weight


class LinearResProp(Function):
    """ Standard forward/backward implementation for a nn.Linear
    layer BUT, uses sparsification as defined in ResProp."""

    # adapted ResProp to linear layers

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        observer,
        th,
        prev_grad_output=None,
        stochastic=True,
        bias=None,
    ):

        ## perform pre-ResProp stage (Algorithm 1)
        preGinput = None
        preGweight = None

        # first we get preGinput by first expanding the previously stored
        # gradients along the batch dimension (replicating w/o copying)
        if prev_grad_output is not None:
            prev_grad_output = (
                torch.unsqueeze(prev_grad_output, 0)
                if stochastic
                else prev_grad_output
            )
            preGinput = prev_grad_output.mm(weight)  # Line 5 in Algo. 1

        # now we get the prGweights by first averaging the inputs across the
        # channel dimension
        if prev_grad_output is not None:
            avg_input = (
                torch.unsqueeze(torch.mean(input, 0), 0)
                if stochastic
                else input
            )
            preGweight = prev_grad_output.t().mm(avg_input)  # Line 4 in Algo. 1

        ctx.save_for_backward(
            input, weight, bias, preGinput, preGweight, prev_grad_output
        )

        ctx.threshold = th
        ctx.observer = observer
        ctx.stch = stochastic
        return nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):

        (
            input,
            weight,
            bias,
            preGinput,
            preGweight,
            prev_grad_output,
        ) = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # compare grad_output to that of in the previous step and threshold
        if prev_grad_output is not None:
            pgo = prev_grad_output
            pg = (
                pgo.expand(input.shape[0], *pgo.size()[1:]) if ctx.stch else pgo
            )
            diff = grad_output - pg  # second term Eq. 4
            grad_output[
                abs(diff) < ctx.threshold
            ] = 0.0  # Applying threshold (Eq. 5)
            non_zero_ratio = grad_output.count_nonzero() / grad_output.numel()
            ctx.observer[1] = 100 - 100 * non_zero_ratio.item()
            # print(f"spHG sparsity: {ctx.threshold[2]:.2f}%")

            # reuse 90% means that we want the resulting `grad_output` to be 90% sparse (i.e. so we keep 90% of the values
            # using the same as in the previous iteration, i.e. `prev_grad_output` )
            if (1.0 - non_zero_ratio) < ctx.observer[0]:
                ctx.threshold.data *= 2.0
            else:
                ctx.threshold.data /= 2.0

        if ctx.needs_input_grad[0]:
            spGrad_input = grad_output.mm(weight)
            if preGinput is not None:
                grad_input = spGrad_input
                if ctx.stch:
                    grad_input += preGinput.expand(
                        spGrad_input.shape[0], *preGinput.size()[1:]
                    )
                else:
                    grad_input += preGinput
            else:
                grad_input = spGrad_input

        if ctx.needs_input_grad[1]:
            spGrad_weight = grad_output.t().mm(input)
            grad_weight = (
                spGrad_weight
                if preGweight is None
                else spGrad_weight + preGweight
            )

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, None, None, None, None, grad_bias


class Conv2dResProp(Function):
    """ Standard forward/backward implementation for a nn.Conv2d
    layer BUT, uses sparsification of gradients (both inputs and weights)
    as described in ResProp. """

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        observer,
        th,
        prev_grad_output=None,
        stochastic=True,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):

        ## perform pre-ResProp stage (Algorithm 1)
        preGinput = None
        preGweight = None

        # obtain preGinput
        if prev_grad_output is not None:
            gA = (
                torch.unsqueeze(prev_grad_output, 0)
                if stochastic
                else prev_grad_output
            )
            preGinput = conv2d_input(
                input.size(), weight, gA, stride, padding, dilation, groups
            )  # Line 5 in Algo. 1

        # now we get the prGweights by first averaging the inputs across the
        # channel dimension
        if prev_grad_output is not None:
            if stochastic:
                gW = torch.unsqueeze(prev_grad_output, 0)
                inpt = torch.unsqueeze(
                    torch.mean(input, 0), 0
                )  # compute average across inputs
            else:
                gW = prev_grad_output
                inpt = input
            preGweight = conv2d_weight(
                inpt, weight.size(), gW, stride, padding, dilation, groups
            )  # Line 4 in Algo. 1

        ctx.save_for_backward(
            input, weight, bias, preGinput, preGweight, prev_grad_output
        )
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.threshold = th
        ctx.observer = observer
        ctx.stch = stochastic
        output = nn.functional.conv2d(
            input, weight, bias, stride, padding, dilation, groups
        )  # Line 3 Algo. 1
        return output

    @staticmethod
    def backward(ctx, grad_output):

        (
            input,
            weight,
            bias,
            preGinput,
            preGweight,
            prev_grad_output,
        ) = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Compute bias gradients with dense grad_output
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((2, 3))

        # compare grad_output to that of in the previous step and threshold
        if prev_grad_output is not None:
            pgo = prev_grad_output
            pg = (
                pgo.view(1, *pgo.size()).expand(input.shape[0], *pgo.size())
                if ctx.stch
                else pgo
            )
            diff = grad_output - pg  # second term Eq. 4
            grad_output[
                abs(diff) < ctx.threshold
            ] = 0.0  # Applying threshold (Eq. 5)
            non_zero_ratio = grad_output.count_nonzero() / grad_output.numel()
            ctx.observer[1] = 100 - 100 * non_zero_ratio.item()
            # print(f"spHG sparsity: {ctx.threshold[2]:.2f}%")

            # reuse 90% means that we want the resulting `grad_output` to be 90% sparse (i.e. so we keep 90% of the values
            # using the same as in the previous iteration, i.e. `prev_grad_output` )
            if (1.0 - non_zero_ratio) < ctx.observer[0]:
                ctx.threshold.data *= 2.0
            else:
                ctx.threshold.data /= 2.0

        spHG = grad_output

        if ctx.needs_input_grad[0]:
            spGrad_input = conv2d_input(
                input.size(),
                weight,
                spHG,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups,
            )
            # Eq 7
            if preGinput is not None:
                # replicate preGinput and add to spGrad_input
                grad_input = spGrad_input
                if ctx.stch:
                    grad_input += preGinput.expand(
                        spGrad_input.shape[0], *preGinput.size()[1:]
                    )
                else:
                    grad_input += preGinput
            else:
                grad_input = spGrad_input

        if ctx.needs_input_grad[1]:
            spGrad_weight = conv2d_weight(
                input,
                weight.size(),
                spHG,
                ctx.stride,
                ctx.padding,
                ctx.dilation,
                ctx.groups,
            )
            # Eq 8 (preGweight is of shape [1,1,k,k] so the sum will be broadcasted)
            grad_weight = (
                spGrad_weight
                if preGweight is None
                else spGrad_weight + preGweight
            )

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
        )


def save_gradients_wrt_output(self, grad_input, grad_output):
    """ Gets called once the gradient w.r.t output. Here we
    save a random sample of `grad_output` for the next training
    step (so we can reuse it)."""

    # Keep gradients from one sample along the batch dimension
    # print(grad_output[0].shape)
    if self.stochastic:
        idx = torch.randint(low=0, high=grad_output[0].shape[0], size=())
        self.prev_grad_output = (
            grad_output[0][idx].detach().clone()
        )  # this is the Radn(dL/dy)_{i-1} in Fig 3
    else:
        self.prev_grad_output = (
            grad_output[0].detach().clone()
        )  # keep the entire grad tensor


class resPropLinear(nn.Linear):
    """ A standard Linear layer using sparse gradient as in ResProp. """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        sparsity: float = 0.5,
        warmup: int = 0,
        stochastic: bool = True,
    ):
        super(resPropLinear, self).__init__(
            input_features, output_features, bias
        )
        # The th list:
        # [0] the target reuse ratio
        # [1] will store the final sparsity obtained (updated with every step)
        self.observer = [sparsity, 0.0]
        # threshold to sparsify diff of gradients from consecutive steps
        self.threshold = torch.nn.Parameter(
            torch.tensor(1e-7), requires_grad=False
        )
        # number of rounds to wait until full target reuse ratio
        self.warmup = warmup
        self.stochastic = stochastic
        self.prev_grad_output = None
        self.register_backward_hook(save_gradients_wrt_output)

    def forward(self, input):
        # print(f"Sparsity (reuse: {100*self.th[0]}% , th: {self.th[1]:.3e}) in previous step: {self.th[2]:.2f} %")
        return LinearResProp.apply(
            input,
            self.weight,
            self.observer,
            self.threshold,
            self.prev_grad_output,
            self.stochastic,
            self.bias,
        )

    def adjust_sparsity(self, current_round: int):
        """ Linearly scales sparsity ratio given the current round """
        # if `current_round` < warmup, then we linearly scale down `target_sp`
        sp_step = self.observer[0] / max(1, self.warmup)

        # set new sparsity/reuse target for this layer
        self.observer[0] = min(sp_step * current_round, self.observer[0])

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class resPropConv2d(nn.Conv2d):
    """ A standard Conv2d layer using sparse gradient as in ResProp. """

    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        sparsity: float = 0.5,
        warmup: int = 0,
        stochastic: bool = True,
    ):
        super(resPropConv2d, self).__init__(
            input_features,
            output_features,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

        # The th list:
        # [0] the target reuse ratio
        # [1] will store the final sparsity obtained (updated with every step)
        self.observer = [sparsity, 0.0]
        # threshold to sparsify diff of gradients from consecutive steps
        self.threshold = torch.nn.Parameter(
            torch.tensor(1e-7), requires_grad=False
        )
        # number of rounds to wait until full target reuse ratio
        self.warmup = warmup
        self.stochastic = stochastic
        self.prev_grad_output = None
        self.register_backward_hook(save_gradients_wrt_output)

    def forward(self, input):
        # print(f"Sparsity (reuse: {100*self.observer[0]}% , th: {self.threshold.item():.3e}) in previous step: {self.observer[1]:.2f} %")
        return Conv2dResProp.apply(
            input,
            self.weight,
            self.observer,
            self.threshold,
            self.prev_grad_output,
            self.stochastic,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def adjust_sparsity(self, current_round: int):
        """ Linearly scales sparsity ratio given the current round """
        # if `current_round` < warmup, then we linearly scale down `target_sp`
        sp_step = self.observer[0] / max(1, self.warmup)

        # set new sparsity/reuse target for this layer
        self.observer[0] = min(sp_step * current_round, self.observer[0])

    def extra_repr(self):
        return "in_channels={}, out_channels={}, bias={}".format(
            self.in_channels, self.out_channels, self.bias is not None,
        )
