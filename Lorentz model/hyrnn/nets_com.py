import itertools
import torch.nn as nn
import torch.nn.functional
import math
import hyperboloid_com as lmath
import sys

def mobius_linear(
    input,
    weight,
    k,
    num_block,
    bias=None,
    hyperbolic_input=False,
    nonlin=None,
    hyperbolic_output=False,
):
    if not hyperbolic_input:
        weight = weight.reshape((-1, num_block, weight.size(-1) // num_block))
        input = input.reshape((-1, num_block, input.size(-1) // num_block))
        input = torch.cat((torch.zeros_like(input.narrow(-1,0,1)), input), -1)
        input = lmath.expmap0(input, k=k)
        output = None
    for blk in range(num_block):
        if output == None:
            output = lmath.mobius_matvec(weight[:,blk], input[:,blk], k=k)
        else:
            output = lmath.mobius_add(output, lmath.mobius_matvec(weight[:,blk], input[:,blk], k=k), k=k)
    assert torch.isfinite(output).all()
    if bias is not None:
        bias = lmath.expmap0(bias, k=k)
        output = lmath.mobius_add(output, bias, k=k) 
    assert torch.isfinite(output).all()
    output = lmath.logmap0(output, k=k)
    if nonlin is not None:
        output = nonlin(output)
    if hyperbolic_output:
        output = lmath.expmap0(output, k=k)
    assert torch.isfinite(output).all()
    output = output.narrow(-1, 1, output.size(-1)-1).sigmoid()
    return output


class MobiusLinear_c(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_block=1,        
        hyperbolic_input=False,
        hyperbolic_output=False,
        nonlin=None
    ):
        super().__init__()
        self.weight = nn.Parameter(1e-2 * torch.randn(out_dim + 1, in_dim + num_block))
        self.num_block = num_block 
        self.bias = nn.Parameter(torch.cat((torch.zeros(1), 1e-4 * torch.randn(out_dim)), dim=0))
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_output = hyperbolic_output
        self.nonlin = nonlin

    def forward(self, input, k):
        return mobius_linear(
            input,
            weight=self.weight,
            k=k,
            num_block=self.num_block,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_output=self.hyperbolic_output,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "hyperbolic_input={}".format(self.hyperbolic_input)
        return info

def mobius_add_chunk(U, x, num_block, k):
    output = None
    U_block = U.chunk(num_block, -1)
    x_block = x.chunk(num_block, -1)

    for blk in range(num_block):
        if output == None:
            output = lmath.mobius_matvec(U_block[blk], x_block[blk], k=k)
        else:
            output = lmath.mobius_add(output, lmath.mobius_matvec(U_block[blk], x_block[blk], k=k), k=k)
    return output


def one_rnn_transform(W, h, U, x, nb, b, k):
    W_otimes_h = lmath.mobius_matvec(W, h, k=k)
    U_otimes_x = mobius_add_chunk(U, x, nb, k=k)
    Wh_plus_Ux = lmath.mobius_add(W_otimes_h, U_otimes_x, k=k)
    return lmath.mobius_add(Wh_plus_Ux, b, k=k)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    num_block: int,
    k: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias.chunk(3)
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = lmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, num_block, b_z, k), k=k).sigmoid()
    r_t = lmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, num_block, b_r, k), k=k).sigmoid()

    rh_t = lmath.expmap0(r_t * lmath.logmap0(hx, k=k), k=k)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, num_block, b_h, k)
    if nonlin is not None:
        h_tilde = lmath.expmap0(nonlin(lmath.logmap0(h_tilde, k=k)), k=k)

    delta_h = lmath.mobius_add(lmath.proj(-hx, k=k), h_tilde, k=k)
    h_int = lmath.ptransp0(hx, z_t * lmath.logmap0(delta_h, k=k), k=k)
    h_out = lmath.expmap(h_int, hx, k=k)

    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    num_block: int,
    k: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    hyperbolic_output: bool = False,
    nonlin=None,
):
    assert bias.isfinite().all()
    bias = lmath.expmap0(bias, k=k)
    if not hyperbolic_hidden_state0:
        hx = lmath.expmap0(h0, k=k)
    else:
        hx = h0
    if not hyperbolic_input:
        if batch_sizes is None:
            input = input.reshape((1, -1, num_block, input.size(-1) // num_block))
            input = torch.cat((torch.zeros_like(input.narrow(-1,0,1)), input), -1)
            input = lmath.expmap0(input, k=k)
            input = input.reshape((1, -1, num_block * input.size(-1)))
        else:
            input = input.reshape((-1, num_block, input.size(-1) // num_block))
            input = torch.cat((torch.zeros_like(input.narrow(-1,0,1)), input), -1)
            input = lmath.expmap0(input, k=k)
            input = input.reshape((-1, num_block * input.size(-1)))
    outs = []
    if batch_sizes is None:
        hx = hx[0]
        input_unbinded = input.unbind(1)
        for t in range(input.size(1)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                num_block=num_block,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                num_block=num_block,
                nonlin=nonlin,
                k=k,
            )
            assert hx.isfinite().all()            
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    assert outs.isfinite().all()
    assert h_last.isfinite().all()
#    print(h_last.isfinite().all())
    if not hyperbolic_output:
        h_last = lmath.logmap0(h_last, k=k)
#        print(h_last.isfinite().all())
#        print(torch.sum(h_last, -1).isfinite().all())
        assert h_last.isfinite().all()
        h_last = h_last.narrow(-1, 1, h_last.size(-1)-1)
    return outs, h_last


class MobiusGRU_c(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        num_block=1,
        bias=True,
        nonlin=None,
        hyperbolic_input=False,
        hyperbolic_hidden_state0=True,
        hyperbolic_output=False
    ):
        super().__init__()
        self.input_size = input_size + num_block
        self.hidden_size = hidden_size + 1
        self.num_block = num_block
        self.num_layers = num_layers
        self.weight_ih = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(3 * self.hidden_size, self.input_size if i == 0 else self.hidden_size))
                for i in range(num_layers)]
        )
        self.weight_hh = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(3 * self.hidden_size, self.hidden_size))
                for _ in range(num_layers)]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.nn.Parameter(torch.cat((torch.zeros(3, 1), torch.randn(3, hidden_size) * 1e-5), dim=-1))
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.hyperbolic_output = hyperbolic_output
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, k, h0=None):
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
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        outputs = []
        last_states = []
        out = input

        assert input.isfinite().all()
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=self.bias[i],
                num_block=self.num_block,
                k=k,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                hyperbolic_output=self.hyperbolic_output,
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
        ).format(**self.__dict__, self=self, bias=self.bias is not None)
