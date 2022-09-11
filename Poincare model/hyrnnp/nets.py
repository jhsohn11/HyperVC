import itertools
import torch.nn as nn
import torch.nn.functional
import math
import geoopt.manifolds.stereographic.math as pmath
import geoopt
from hyrnnp.k_utils import *

            
def mobius_linear(
    input,
    weight,
    k,
    bias=None,
    hyperbolic_input=False,
    hyperbolic_bias=False,
    nonlin=None,
    hyperbolic_output=False,
):
    if torch.cuda.is_available():
        k = k.to('cuda:0')
    if hyperbolic_input:
        output = mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = expmap0(output, k=k)
    assert not torch.isnan(output).any()
    if bias is not None:
        if not hyperbolic_bias:
            bias = expmap0(bias, k=k)
        output = mobius_add(output, bias, k=k)
    assert not torch.isnan(output).any()
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, k=k)
    output = project(output, k=k)
    assert not torch.isnan(output).any()
    if not hyperbolic_output:
        output = logmap0(output, k=k)
    return output


class MobiusLinear_d(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=False,
        hyperbolic_bias=False,
        nonlin=None,
        hyperbolic_output=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
            self.bias.normal_() / 4
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_output = hyperbolic_output
        self.nonlin = nonlin

    def forward(self, input, k):
        return mobius_linear(
            input,
            weight=self.weight,
            k=k,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            hyperbolic_output=self.hyperbolic_output
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info



def one_rnn_transform(W, h, U, x, b, k):
    W_otimes_h = mobius_matvec(W, h, k=k)
    U_otimes_x = mobius_matvec(U, x, k=k)
    Wh_plus_Ux = mobius_add(W_otimes_h, U_otimes_x, k=k)
    return mobius_add(Wh_plus_Ux, b, k=k)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    k: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, k), k=k).sigmoid()
    r_t = logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, k), k=k).sigmoid()

    rh_t = mobius_pointwise_mul(r_t, hx, k=k)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, k)

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, k=k)
    delta_h = mobius_add(-hx, h_tilde, k=k)
    h_out = mobius_add(hx, mobius_pointwise_mul(z_t, delta_h, k=k), k=k)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    k: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_input:
        input = expmap0(input, k=k)
    outs = []
    if batch_sizes is None:
        hx = h0[0]
        input_unbinded = input.unbind(1)
        for t in range(input.size(1)):
            hx = expmap0(hx, k=k)
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            hx = logmap0(hx, k=k)
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        hx = h0
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            kx, k = k[:batch_sizes[t]], k[batch_sizes[t]:]
            hx = expmap0(hx, k=kx)
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=kx,
            )
            hx = logmap0(hx, k=kx)
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


class MobiusGRU_d(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlin=None,
        hyperbolic_input=False,
        hyperbolic_hidden_state0=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.k_biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(3, hidden_size) * 1e-5) for _ in range(num_layers)]
        )
        
        self.weight_ih = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)) for i in range(num_layers)]
        )
        self.weight_hh = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size)) for _ in range(num_layers)]
        )
        if bias:
            self.bias = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(3, hidden_size) * 1e-5) for _ in range(num_layers)]
                )
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, k: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, max_batch_size, self.hidden_size, requires_grad=False).cuda()
        h0 = h0.unbind(0)
        
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input

        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                k=k,
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht  

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "k={self.k}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)
        
        
#class MobiusDist2Hyperplane(torch.nn.Module):
#    def __init__(self, in_features, out_features, c=torch.tensor(0.5)):
#        super().__init__()
#        self.in_features = in_features
#        self.out_features = out_features
#        self.ball = ball = geoopt.PoincareBall(c=c)
#        self.sphere = sphere = geoopt.manifolds.Sphere()
#        self.scale = torch.nn.Parameter(torch.zeros(out_features))
#        point = torch.randn(out_features, in_features) / 4
#        point = pmath.expmap0(point, k=k)
#        tangent = torch.randn(out_features, in_features)
#        self.point = geoopt.ManifoldParameter(point, manifold=ball)
#        with torch.no_grad():
#            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()
#
#    def forward(self, input):
#        input = input.unsqueeze(-2)
#        distance = pmath.dist2plane(
#            x=input, p=self.point, a=self.tangent, c=self.ball.c, signed=True
#        )
#        return distance * self.scale.exp()
#
#    def extra_repr(self):
#        return (
#            "in_features={in_features}, out_features={out_features}, "
#            "c={ball.c}".format(
#                **self.__dict__
#            )
#        )
#