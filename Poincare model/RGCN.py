import torch
import torch.nn as nn
import dgl.function as fn
import geoopt.manifolds.stereographic.math as pmath

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, k, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        self.k = k

    # define how propagation is done in subclass
    def propagate(self, g, reverse):
        raise NotImplementedError

    def forward(self, g, reverse):
        k = self.k.cuda()
        if self.self_loop:
            loop_message = pmath.mobius_matvec(self.loop_weight, g.ndata['h'], k=k)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        self.propagate(g, reverse)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = pmath.mobius_add(node_repr, self.bias, k=k)
        if self.self_loop:
            node_repr = pmath.mobius_add(node_repr, loop_message, k=k)
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return g


class HyperbolicRGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, k, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(HyperbolicRGCNBlockLayer, self).__init__(in_feat, out_feat, k, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0
        
        self.k = k

        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        # if self.num_rels == 2:
        #     self.in_feat = in_feat
        #     self.weight = nn.Parameter(torch.Tensor(
        #         self.num_rels, in_feat, out_feat))
        # else:
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges, reverse):
        k=self.k.cuda()
        if reverse:
            weight = self.weight.index_select(0, edges.data['type_o']).view(
                -1, self.submat_in, self.submat_out)
        else:
            weight = self.weight.index_select(0, edges.data['type_s']).view(
                        -1, self.submat_in, self.submat_out)
        node = edges.src['h']
        node = pmath.logmap0(node, k=k).view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        msg = pmath.expmap0(msg, k=k)
        
        return {'msg': msg}

    def propagate(self, g, reverse):
        g.update_all(lambda x: self.msg_func(x, reverse), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
