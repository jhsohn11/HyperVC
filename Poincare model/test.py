import argparse
import numpy as np
import torch
import utils
import os
from model import HyperbolicRENet
from global_model import HyperbolicRENet_global
import pickle


def test(args):
    # load data
    num_nodes, num_rels = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    if args.dataset == 'ICEWS14':
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'test0.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
        t_hscore = utils.get_hierarchical_score('./data/' + args.dataset, 'train_hierarchy.txt', 'test_hierarchy.txt')
    else:
        train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
        valid_data, valid_times = utils.load_quadruples('./data/' + args.dataset, 'valid00.txt')
        test_data, test_times = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
        total_data, total_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
        t_hscore = utils.get_hierarchical_score('./data/' + args.dataset, 'train_hierarchy.txt', 'valid_hierarchy.txt', 'test_hierarchy.txt')
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(999)


    model_state_file = 'models/' + args.dataset + '/gat.pth'
    model_graph_file = 'models/' + args.dataset + '/gat_graph.pth'
    model_state_global_file2 = 'models/' + args.dataset + '/max' + str(args.maxpool) + 'gat2_global.pth'
    model_state_global_file = 'models/' + args.dataset + '/max' + str(args.maxpool) + 'gat_global.pth'

    model = HyperbolicRENet(num_nodes,
                    args.n_hidden,
                    num_rels,
                    t_hscore,
                    model=args.model,
                    seq_len=args.seq_len,
                    num_k=args.num_k)
    global_model = HyperbolicRENet_global(num_nodes,
                                args.n_hidden,
                                num_rels,
                                args.k,
                                model=args.model,
                                seq_len=args.seq_len,
                                num_k=args.num_k, maxpool=args.maxpool)


    if use_cuda:
        model.cuda()
        global_model.cuda()


    with open('data/' + args.dataset+'/test_history_sub.txt', 'rb') as f:
        s_history_test_data = pickle.load(f)
    with open('data/' + args.dataset+'/test_history_ob.txt', 'rb') as f:
        o_history_test_data = pickle.load(f)

    s_history_test = s_history_test_data[0]
    s_history_test_t = s_history_test_data[1]
    o_history_test = o_history_test_data[0]
    o_history_test_t = o_history_test_data[1]



    print("start testing:")
        
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.s_hist_test = checkpoint['s_hist']
    model.s_his_cache = checkpoint['s_cache']
    model.o_hist_test = checkpoint['o_hist']
    model.o_his_cache = checkpoint['o_cache']
    model.latest_time = checkpoint['latest_time']
    if args.dataset == "ICEWS14":
        model.latest_time = torch.LongTensor([4344])[0]
    model.global_emb = checkpoint['global_emb']
    model.global_k = checkpoint['global_k']
    model.s_hist_test_t = checkpoint['s_hist_t']
    model.s_his_cache_t = checkpoint['s_cache_t']
    model.o_hist_test_t = checkpoint['o_hist_t']
    model.o_his_cache_t = checkpoint['o_cache_t']
    with open(model_graph_file, 'rb') as f:
        model.graph_dict = pickle.load(f)

    checkpoint_global2 = torch.load(model_state_global_file2, map_location=lambda storage, loc: storage)
    global_model.load_state_dict(checkpoint_global2['state_dict'])
    global_model.global_emb = checkpoint_global2['global_emb']
    global_model.global_k = checkpoint['global_k']


    print("Using best epoch: {}".format(checkpoint['epoch']))

    softplus = torch.nn.Softplus()
    print("Global curvature: {}".format(global_model.global_k.data[0]))
    print("layers1_weight: {}, layers1_bias: {}".format(model.k_hs.linear1_k.weight.data[0][0], model.k_hs.linear1_k.bias.data[0]))
    print("layers2_weight: {}, layers1_bias: {}".format(model.k_hs.linear2_k.weight.data[0][0], model.k_hs.linear2_k.bias.data[0]))

    total_data = torch.from_numpy(total_data)
    test_data = torch.from_numpy(test_data)

    model.eval()
    global_model.eval()
    total_loss = 0
    total_ranks = np.array([])
    total_ranks_filter = np.array([])
    ranks = []
    for ee in range(num_nodes):
        while len(model.s_hist_test[ee]) > args.seq_len:
            model.s_hist_test[ee].pop(0)
            model.s_hist_test_t[ee].pop(0)
        while len(model.o_hist_test[ee]) > args.seq_len:
            model.o_hist_test[ee].pop(0)
            model.o_hist_test_t[ee].pop(0)

    if use_cuda:
        total_data = total_data.cuda()
        
    latest_time = test_times[0]
    for i in range(len(test_data)):
        batch_data = test_data[i]
        s_hist = s_history_test[i]
        o_hist = o_history_test[i]
        s_hist_t = s_history_test_t[i]
        o_hist_t = o_history_test_t[i]
        if latest_time != batch_data[3]:
            ranks.append(total_ranks_filter)
            latest_time = batch_data[3]
            total_ranks_filter = np.array([])

        if use_cuda:
            batch_data = batch_data.cuda()

        with torch.no_grad():
            # Filtered metric
            ranks_filter, loss = model.evaluate_filter(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t), global_model, total_data)
            total_ranks_filter = np.concatenate((total_ranks_filter, ranks_filter))
            total_loss += loss.item()

    ranks.append(total_ranks_filter)

    for rank in ranks:
        total_ranks = np.concatenate((total_ranks,rank))
    mrr = np.mean(1.0 / total_ranks)
    mr = np.mean(total_ranks)
    hits = []

    for hit in [1,3,10]:
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count))
    print("MRR (filtered): {:.6f}".format(mrr))
    print("MR (filtered): {:.6f}".format(mr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18',
            help="dataset to use")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--model", type=int, default=3)
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=1000,
                    help="cuttoff position")
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument("--k", type=float, default=0.0, help="starting curvature")

    args = parser.parse_args()
    test(args)

