import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from dgl.nn.pytorch import GATConv

class GATAggregator_global(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, model, seq_len=10, maxpool=1, num_heads=3):
        super(GATAggregator_global, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.model = model
        self.maxpool = maxpool
        
        self.gat1 =  GATConv(h_dim, h_dim, num_heads)
        self.gat2 =  GATConv(h_dim, h_dim, num_heads)


    def forward(self, t_list, global_embeds, graph_dict, reverse):
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        num_non_zero = len(torch.nonzero(t_list))
        t_list = t_list[:num_non_zero]

        for tim in t_list:
            length = int(tim // time_unit)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        time_to_idx = dict()
        g_list = []
        idx = 0
        for tim in unique_t:
            time_to_idx[tim.item()] = idx
            idx += 1
            g_list.append(graph_dict[tim.item()])

        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(torch.device('cuda:0'))
        batched_graph.ndata['h'] = global_embeds.weight[batched_graph.ndata['id']].view(-1, self.h_dim)
        batched_graph.ndata['h'] = self.gat1(batched_graph, batched_graph.ndata['h']).mean(dim=1)
        batched_graph.ndata['h'] = self.gat2(batched_graph, batched_graph.ndata['h']).mean(dim=1)

        if self.maxpool == 1:
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')

        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim).cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_info[time_to_idx[t.item()]]

        embed_seq_tensor = self.dropout(embed_seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        return packed_input

    def predict(self, t, global_embeds, graph_dict, reverse):
        times = list(graph_dict.keys())

        id = 0
        for tt in times:
            if tt >= t:
                break
            id += 1

        if self.seq_len <= id:
            timess = torch.LongTensor(times[id - self.seq_len:id])
        else:
            timess = torch.LongTensor(times[:id])

        g_list = []

        for tim in timess:
            graph_cuda = graph_dict[tim.item()].to(torch.device('cuda:0'))
            g_list.append(graph_cuda)

        batched_graph = dgl.batch(g_list)
        batched_graph = batched_graph.to(torch.device('cuda:0'))
        batched_graph.ndata['h'] = global_embeds.weight[batched_graph.ndata['id']].view(-1, self.h_dim)
        batched_graph.ndata['h'] = self.gat1(batched_graph, batched_graph.ndata['h']).mean(dim=1)
        batched_graph.ndata['h'] = self.gat2(batched_graph, batched_graph.ndata['h']).mean(dim=1)

        if self.maxpool == 1:
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')
        
        return global_info


class GATAggregator(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, model, seq_len=10, num_heads=3):
        super(GATAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.model = model

        self.gat1 =  GATConv(h_dim, h_dim, num_heads, allow_zero_in_degree=True)
        self.gat2 =  GATConv(h_dim, h_dim, num_heads, allow_zero_in_degree=True)

    def forward(self, s_hist, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, reverse):
        length = 0
        for his in s_hist[0]:
            length += len(his)
        if length == 0:
            s_packed_input = None
        else:
            s_len_non_zero, s_hist_t_sorted, s_tem, r_tem, g, node_ids_graph, global_emb_list = get_sorted_s_r_embed_rgcn(s_hist, s, r, ent_embeds, graph_dict, global_emb)
            if g is None:
                s_packed_input = None
            else:

                g.ndata['h'] = self.gat1(g, g.ndata['h']).mean(dim=1)
                g.ndata['h'] = self.gat2(g, g.ndata['h']).mean(dim=1)

                embeds_mean = g.ndata.pop('h')
                embeds_mean = embeds_mean[torch.LongTensor(node_ids_graph)]
                embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())
                global_emb_list_split = torch.split(global_emb_list, s_len_non_zero.tolist())
                
                s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 4 * self.h_dim).cuda()
                s_embed_seq_tensor_r = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()
                s_embed_seq_t = torch.zeros(len(s_len_non_zero), self.seq_len).cuda()

                # Slow!!!
                for i, embeds in enumerate(embeds_split):
                    s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                         rel_embeds[r_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)
                    s_embed_seq_tensor_r[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)
                    s_embed_seq_t[i, torch.arange(len(embeds))] = torch.Tensor(s_hist_t_sorted[i]).cuda()

                s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
                s_embed_seq_tensor_r = self.dropout(s_embed_seq_tensor_r)

                s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                         s_len_non_zero,
                                                                         batch_first=True)
                s_packed_input_r = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor_r,
                                                                           s_len_non_zero,
                                                                           batch_first=True)
                s_packed_t = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_t,
                                                                     s_len_non_zero,
                                                                     batch_first=True)

        return s_packed_input, s_packed_input_r, s_packed_t

    def predict_batch(self, s_hist, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, reverse):
        length = 0
        for his in s_hist[0]:
            length += len(his)
        if length == 0:
            s_packed_input = None
            s_packed_input_r = None
        else:

            s_len_non_zero, s_hist_t_sorted, s_tem, r_tem, g, node_ids_graph, global_emb_list = get_s_r_embed_rgcn(s_hist, s, r, ent_embeds, graph_dict, global_emb)
            if g is None:
                s_packed_input = None
            else:

                g.ndata['h'] = self.gat1(g, g.ndata['h']).mean(dim=1)
                g.ndata['h'] = self.gat2(g, g.ndata['h']).mean(dim=1)

                embeds_mean = g.ndata.pop('h')
                embeds_mean = embeds_mean[torch.LongTensor(node_ids_graph)]
                embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())
                global_emb_list_split = torch.split(global_emb_list, s_len_non_zero.tolist())

                s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 4 * self.h_dim).cuda()
                s_embed_seq_tensor_r = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()
                s_embed_seq_t = torch.zeros(len(s_len_non_zero), self.seq_len).cuda()

                # Slow!!!
                for i, embeds in enumerate(embeds_split):
                    s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                         rel_embeds[r_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)
                    s_embed_seq_tensor_r[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)
                    s_embed_seq_t[i, torch.arange(len(embeds))] = torch.Tensor(s_hist_t_sorted[i]).cuda()

                s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
                s_embed_seq_tensor_r = self.dropout(s_embed_seq_tensor_r)

                s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                         s_len_non_zero,
                                                                         batch_first=True)
                s_packed_input_r = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor_r,
                                                                           s_len_non_zero,
                                                                           batch_first=True)
                s_packed_t = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_t,
                                                                     s_len_non_zero,
                                                                     batch_first=True)

        return s_packed_input, s_packed_input_r, s_packed_t



    def predict(self, s_history, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, reverse):
        s_hist = s_history[0]
        s_hist_t = s_history[1]
        s_len_non_zero, s_hist_t_sorted, s_tem, r_tem, g, node_ids_graph, global_emb_list = get_s_r_embed_rgcn(([s_hist], [s_hist_t]), s.view(-1,1), r.view(-1,1), ent_embeds,
                                                                                      graph_dict, global_emb)

        g.ndata['h'] = self.gat1(g, g.ndata['h']).mean(dim=1)
        g.ndata['h'] = self.gat2(g, g.ndata['h']).mean(dim=1)

        embeds_mean = g.ndata.pop('h')
        embeds = embeds_mean[torch.LongTensor(node_ids_graph)]

        inp = torch.zeros(len(s_hist), 4 * self.h_dim).cuda()
        inp[torch.arange(len(embeds)), :] = torch.cat(
            (embeds, ent_embeds[s].repeat(len(embeds), 1), rel_embeds[r].repeat(len(embeds), 1), global_emb_list), dim=1)

        inp_r = torch.zeros(len(s_hist), 3 * self.h_dim).cuda()
        inp_r[torch.arange(len(embeds)), :] = torch.cat((embeds, ent_embeds[s].repeat(len(embeds), 1), global_emb_list), dim=1)
        
        inp_t = torch.Tensor(s_hist_t).cuda()

        return inp, inp_r, inp_t

