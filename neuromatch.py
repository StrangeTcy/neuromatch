#!g1.1
"""Train the order embedding model"""

# import faulthandler

# faulthandler.enable()


# Set this flag to True to use hyperparameter optimization
# We use Testtube for hyperparameter tuning
HYPERPARAM_SEARCH = False
HYPERPARAM_SEARCH_N_TRIALS = None   # how many grid search trials to run
                                    #    (set to None for exhaustive search)

import argparse
from email.policy import default
from itertools import permutations
from multiprocessing.dummy import freeze_support
import pickle
import queue
from queue import PriorityQueue
import os
import random
import time


import networkx as nx
import numpy as np
import logging
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm, trange

from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
import orca
from torch_scatter import scatter_add


from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import TUDataset

from functools import reduce


from deepsnap.dataset import Generator
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import scipy.stats as stats

from collections import defaultdict, Counter

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch


from itertools import permutations


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import wandb

# from common import data


import horovod.torch as hvd


# from common import combined_syn

# Combination of synthetic graph generators # (first used in subgraph matching and motif mining)



import deepsnap.dataset as dataset

class ERGenerator(dataset.Generator):
    def __init__(self, sizes, p_alpha=1.3, **kwargs):
        super(ERGenerator, self).__init__(sizes, **kwargs)
        self.p_alpha = p_alpha

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        # p follows beta distribution with mean = log2(num_graphs) / num_graphs
        alpha = self.p_alpha
        mean = np.log2(num_nodes) / num_nodes
        beta = alpha / mean - alpha
        p = np.random.beta(alpha, beta)
        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            p = np.random.beta(alpha, beta)
            graph = nx.gnp_random_graph(num_nodes, p)
        logging.debug('Generated {}-node E-R graphs with average p: {}'.format(
                num_nodes, mean))
        return graph

class WSGenerator(dataset.Generator):
    def __init__(self, sizes, density_alpha=1.3, 
            rewire_alpha=2, rewire_beta=2, **kwargs):
        super(WSGenerator, self).__init__(sizes, **kwargs)
        self.density_alpha = density_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta
    
    def generate(self, size=None):
        num_nodes = self._get_size(size)
        curr_num_graphs = 0

        density_alpha = self.density_alpha
        density_mean = np.log2(num_nodes) / num_nodes
        density_beta = density_alpha / density_mean - density_alpha

        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta
        while curr_num_graphs < 1:
            k = int(np.random.beta(density_alpha, density_beta) * num_nodes)
            k = max(k, 2)
            p = np.random.beta(rewire_alpha, rewire_beta)
            try:
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
                curr_num_graphs += 1
            except:
                pass
        logging.debug('Generated {}-node W-S graph with average density: {}'.format(
                num_nodes, density_mean))
        return graph

class BAGenerator(dataset.Generator):
    def __init__(self, sizes, max_p=0.2, max_q=0.2, **kwargs):
        super(BAGenerator, self).__init__(sizes, **kwargs)
        self.max_p = 0.2
        self.max_q = 0.2

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        found = False
        m = np.random.choice(max_m) + 1
        p = np.min([np.random.exponential(20), self.max_p])
        q = np.min([np.random.exponential(20), self.max_q])
        while not found:
            graph = nx.extended_barabasi_albert_graph(num_nodes, m, p, q)
            if nx.is_connected(graph):
                found = True
        logging.debug('Generated {}-node extended B-A graph with max m: {}'.format(
                num_nodes, max_m))
        return graph

class PowerLawClusterGenerator(dataset.Generator):
    def __init__(self, sizes, max_triangle_prob=0.5, **kwargs):
        super(PowerLawClusterGenerator, self).__init__(sizes, **kwargs)
        self.max_triangle_prob = max_triangle_prob

    def generate(self, size=None):
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        m = np.random.choice(max_m) + 1
        p = np.random.uniform(high=self.max_triangle_prob)
        found = False
        while not found:
            graph = nx.powerlaw_cluster_graph(num_nodes, m, p)
            if nx.is_connected(graph):
                found = True
        logging.debug('Generated {}-node powerlaw cluster graph with max m: {}'.format(
                num_nodes, max_m))
        return graph

def get_generator(sizes, size_prob=None, dataset_len=None):
    #gen_prob = [1/3.5, 1/3.5, 1/3.5, 0.5/3.5]
    generator = dataset.EnsembleGenerator(
        [ERGenerator(sizes, size_prob=size_prob),
            WSGenerator(sizes, size_prob=size_prob),
            BAGenerator(sizes, size_prob=size_prob),
            PowerLawClusterGenerator(sizes, size_prob=size_prob)],
        #gen_prob=gen_prob,
        dataset_len=dataset_len)
    #print(generator)
    return generator

def get_dataset(task, dataset_len, sizes, size_prob=None, **kwargs):
    generator = get_generator(sizes, size_prob=size_prob,
        dataset_len=dataset_len)
    return dataset.GraphDataset(
        None, task=task, generator=generator, **kwargs)

def main():
    sizes = np.arange(6, 31)
    dataset = get_dataset("graph", sizes)
    print('On the fly generated dataset has length: {}'.format(len(dataset)))
    example_graph = dataset[0]
    print('Example graph: nodes {}; edges {}'.format(example_graph.G.nodes, example_graph.G.edges))

    print('Even the same index causes a new graph to be generated: edges {}'.format(
            dataset[0].G.edges))

    print('This generator has no label: {}, '
          '(but can be augmented via apply_transform)'.format(dataset.num_node_labels))



# from common import feature_preprocess


AUGMENT_METHOD = "concat"
FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = [], []
#FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = ["identity"], [4]
#FEATURE_AUGMENT = ["motif_counts"]
#FEATURE_AUGMENT_DIMS = [73]
#FEATURE_AUGMENT_DIMS = [15]

def norm(edge_index, num_nodes, edge_weight=None, improved=False,
         dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def compute_identity(edge_index, n, k):
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                             device=edge_index.device)
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, 1, n)
    adj_sparse = torch.sparse.FloatTensor(edge_index, edge_weight,
        torch.Size([n, n]))
    adj = adj_sparse.to_dense()

    deg = torch.diag(torch.sum(adj, -1))
    deg_inv_sqrt = deg.pow(-0.5)
    adj = deg_inv_sqrt @ adj @ deg_inv_sqrt 

    diag_all = [torch.diag(adj)]
    adj_power = adj
    for i in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all

class FeatureAugment(nn.Module):
    def __init__(self):
        super(FeatureAugment, self).__init__()

        def degree_fun(graph, feature_dim):
            graph.node_degree = self._one_hot_tensor(
                [d for _, d in graph.G.degree()],
                one_hot_dim=feature_dim)
            return graph

        def centrality_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            centrality = nx.betweenness_centrality(graph.G)
            graph.betweenness_centrality = torch.tensor(
                [centrality[x] for x in
                nodes]).unsqueeze(1)
            return graph

        def path_len_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            graph.path_len = self._one_hot_tensor(
                [np.mean(list(nx.shortest_path_length(graph.G,
                    source=x).values())) for x in nodes],
                one_hot_dim=feature_dim)
            return graph

        def pagerank_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            pagerank = nx.pagerank(graph.G)
            graph.pagerank = torch.tensor([pagerank[x] for x in
                nodes]).unsqueeze(1)
            return graph

        def identity_fun(graph, feature_dim):
            graph.identity = compute_identity(
                graph.edge_index, graph.num_nodes, feature_dim)
            return graph

        def clustering_coefficient_fun(graph, feature_dim):
            node_cc = list(nx.clustering(graph.G).values())
            if feature_dim == 1:
                graph.node_clustering_coefficient = torch.tensor(
                        node_cc, dtype=torch.float).unsqueeze(1)
            else:
                graph.node_clustering_coefficient = FeatureAugment._bin_features(
                        node_cc, feature_dim=feature_dim)

        def motif_counts_fun(graph, feature_dim):
            assert feature_dim % 73 == 0
            counts = orca.orbit_counts("node", 5, graph.G)
            counts = [[np.log(c) if c > 0 else -1.0 for c in l] for l in counts]
            counts = torch.tensor(counts).type(torch.float)
            #counts = FeatureAugment._wave_features(counts,
            #    feature_dim=feature_dim // 73)
            graph.motif_counts = counts
            return graph

        def node_features_base_fun(graph, feature_dim):
            for v in graph.G.nodes:
                if "node_feature" not in graph.G.nodes[v]:
                    graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
            return graph

        self.node_features_base_fun = node_features_base_fun

        self.node_feature_funs = {"node_degree": degree_fun,
            "betweenness_centrality": centrality_fun,
            "path_len": path_len_fun,
            "pagerank": pagerank_fun,
            'node_clustering_coefficient': clustering_coefficient_fun,
            "motif_counts": motif_counts_fun,
            "identity": identity_fun}

    def register_feature_fun(name, feature_fun):
        self.node_feature_funs[name] = feature_fun

    @staticmethod
    def _wave_features(list_scalars, feature_dim=4, scale=10000):
        pos = np.array(list_scalars)
        if len(pos.shape) == 1:
            pos = pos[:,np.newaxis]
        batch_size, n_feats = pos.shape
        pos = pos.reshape(-1)
        
        rng = np.arange(0, feature_dim // 2).astype(
            np.float) / (feature_dim // 2)
        sins = np.sin(pos[:,np.newaxis] / scale**rng[np.newaxis,:])
        coss = np.cos(pos[:,np.newaxis] / scale**rng[np.newaxis,:])
        m = np.concatenate((coss, sins), axis=-1)
        m = m.reshape(batch_size, -1).astype(np.float)
        m = torch.from_numpy(m).type(torch.float)
        return m

    @staticmethod
    def _bin_features(list_scalars, feature_dim=2):
        arr = np.array(list_scalars)
        min_val, max_val = np.min(arr), np.max(arr)
        bins = np.linspace(min_val, max_val, num=feature_dim)
        feat = np.digitize(arr, bins) - 1
        assert np.min(feat) == 0
        assert np.max(feat) == feature_dim - 1
        return FeatureAugment._one_hot_tensor(feat, one_hot_dim=feature_dim)

    @staticmethod
    def _one_hot_tensor(list_scalars, one_hot_dim=1):
        if not isinstance(list_scalars, list) and not list_scalars.ndim == 1:
            raise ValueError("input to _one_hot_tensor must be 1-D list")
        vals = torch.LongTensor(list_scalars).view(-1,1)
        vals = vals - min(vals)
        vals = torch.min(vals, torch.tensor(one_hot_dim - 1))
        vals = torch.max(vals, torch.tensor(0))
        one_hot = torch.zeros(len(list_scalars), one_hot_dim)
        one_hot.scatter_(1, vals, 1.0)
        return one_hot

    def augment(self, dataset):
        dataset = dataset.apply_transform(self.node_features_base_fun,
            feature_dim=1)
        for key, dim in zip(FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS):
            dataset = dataset.apply_transform(self.node_feature_funs[key], 
                feature_dim=dim)
        return dataset

class Preprocess(nn.Module):
    def __init__(self, dim_in):
        super(Preprocess, self).__init__()
        self.dim_in = dim_in
        if AUGMENT_METHOD == 'add':
            self.module_dict = {
                    key: nn.Linear(aug_dim, dim_in)
                    for key, aug_dim in zip(FEATURE_AUGMENT, 
                                            FEATURE_AUGMENT_DIMS)
                    }

    @property
    def dim_out(self):
        if AUGMENT_METHOD == 'concat':
            return self.dim_in + sum(
                    [aug_dim for aug_dim in FEATURE_AUGMENT_DIMS])
        elif AUGMENT_METHOD == 'add':
            return dim_in
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                    AUGMENT_METHOD))

    def forward(self, batch):
        if AUGMENT_METHOD == 'concat':
            feature_list = [batch.node_feature]
            for key in FEATURE_AUGMENT:
                feature_list.append(batch[key])
            batch.node_feature = torch.cat(feature_list, dim=-1)
        elif AUGMENT_METHOD == 'add':
            for key in FEATURE_AUGMENT:
                batch.node_feature = batch.node_feature + self.module_dict[key](
                        batch[key])
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                    AUGMENT_METHOD))
        return batch



def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name="AIDS")
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v1", name="DBLP_v1")
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name: del graph.name
                graph = pyg_utils.to_networkx(graph).to_undirected()
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task

class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError

class OTFSynDataSource(DataSource):
    """ On-the-fly generated synthetic data for training the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    with a pre-defined generator (see combined_syn.py).

    DeepSNAP transforms are used to generate the positive and negative examples.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.generator = get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            dataset = get_dataset("graph", size // 2,
                np.arange(self.min_size + 1, self.max_size + 1))
            sampler = torch.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()) if \
                    use_distributed_sampling else None
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=sampler, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
        train):
        def sample_subgraph(graph, offset=0, use_precomp_sizes=False,
            filter_negs=False, supersample_small_graphs=False, neg_target=None,
            hard_neg_idxs=None):
            if neg_target is not None: graph_idx = graph.G.graph["idx"]
            use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"]
                in hard_neg_idxs)
            done = False
            n_tries = 0
            while not done:
                if use_precomp_sizes:
                    size = graph.G.graph["subgraph_size"]
                else:
                    if train and supersample_small_graphs:
                        sizes = np.arange(self.min_size + offset,
                            len(graph.G) + offset)
                        ps = (sizes - self.min_size + 2) ** (-1.1)
                        ps /= ps.sum()
                        size = stats.rv_discrete(values=(sizes, ps)).rvs()
                    else:
                        d = 1 if train else 0
                        size = random.randint(self.min_size + offset - d,
                            len(graph.G) - 1 + offset)
                start_node = random.choice(list(graph.G.nodes))
                neigh = [start_node]
                frontier = list(set(graph.G.neighbors(start_node)) - set(neigh))
                visited = set([start_node])
                while len(neigh) < size:
                    new_node = random.choice(list(frontier))
                    assert new_node not in neigh
                    neigh.append(new_node)
                    visited.add(new_node)
                    frontier += list(graph.G.neighbors(new_node))
                    frontier = [x for x in frontier if x not in visited]
                if self.node_anchored:
                    anchor = neigh[0]
                    for v in graph.G.nodes:
                        graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                            anchor == v else torch.zeros(1))
                        #print(v, graph.G.nodes[v]["node_feature"])
                neigh = graph.G.subgraph(neigh)
                if use_hard_neg and train:
                    neigh = neigh.copy()
                    if random.random() < 1.0 or not self.node_anchored: # add edges
                        non_edges = list(nx.non_edges(neigh))
                        if len(non_edges) > 0:
                            for u, v in random.sample(non_edges, random.randint(1,
                                min(len(non_edges), 5))):
                                neigh.add_edge(u, v)
                    else:                         # perturb anchor
                        anchor = random.choice(list(neigh.nodes))
                        for v in neigh.nodes:
                            neigh.nodes[v]["node_feature"] = (torch.ones(1) if
                                anchor == v else torch.zeros(1))

                if (filter_negs and train and len(neigh) <= 6 and neg_target is
                    not None):
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        neg_target[graph_idx], neigh)
                    if not matcher.subgraph_is_isomorphic(): done = True
                else:
                    done = True

            return graph, DSGraph(neigh)

        augmenter = FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)
        neg_target = batch_neg_target
        # TODO: use hard negs
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
            int(len(neg_target.G) * 1/2)))
        #hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                if i not in hard_neg_idxs else g)
                for i, g in enumerate(neg_target.G)])
        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
            hard_neg_idxs=hard_neg_idxs)
        if self.node_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                            else torch.zeros(1))
                return g
            neg_target = neg_target.apply_transform(add_anchor)
        pos_target = augmenter.augment(pos_target).to(get_device())
        pos_query = augmenter.augment(pos_query).to(get_device())
        neg_target = augmenter.augment(neg_target).to(get_device())
        neg_query = augmenter.augment(neg_query).to(get_device())
        #print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query

class OTFSynImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly synthetic data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}".format(str(self.node_anchored),
            self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = batch_nx_graphs(pos_a)
            pos_b = batch_nx_graphs(pos_b)
        neg_a = batch_nx_graphs(neg_a)
        neg_b = batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b

class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """
    def __init__(self, dataset_name, node_anchored=False, min_size=5,
        max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=15, min_size=5, seed=None,
        filter_negs=False, sample_method="tree-pair"):
        batch_size = a
        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set
        if seed is not None:
            random.seed(seed)

        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph, a = sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = sample_neigh([graph], random.randint(min_size,
                    len(graph) - 1))
            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)

        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = sample_neigh(graphs, size)
                graph_b, b = sample_neigh(graphs, random.randint(min_size,
                    size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = sample_neigh(graphs, random.randint(min_size,
                    len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic(): # a <= b (b is subgraph of a)
                    continue
            neg_a.append(neigh_a)
            neg_b.append(neigh_b)

        pos_a = batch_nx_graphs(pos_a, anchors=pos_a_anchors if
            self.node_anchored else None)
        pos_b = batch_nx_graphs(pos_b, anchors=pos_b_anchors if
            self.node_anchored else None)
        neg_a = batch_nx_graphs(neg_a, anchors=neg_a_anchors if
            self.node_anchored else None)
        neg_b = batch_nx_graphs(neg_b, anchors=neg_b_anchors if
            self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b

class DiskImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly real data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, dataset_name, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0
        self.dataset = load_dataset(dataset_name)
        self.train_set, self.test_set, _ = self.dataset
        self.dataset_name = dataset_name

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            neighs = []
            for j in range(size // 2):
                graph, neigh = sample_neigh(self.train_set if train else
                    self.test_set, random.randint(self.min_size, self.max_size))
                neighs.append(graph.subgraph(neigh))
            dataset = GraphDataset(neighs)
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=None, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}-{}".format(self.dataset_name.lower(),
            str(self.node_anchored), self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = batch_nx_graphs(pos_a)
            pos_b = batch_nx_graphs(pos_b)
        neg_a = batch_nx_graphs(neg_a)
        neg_b = batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b

def datasets_main():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    for name in ["enzymes", "reddit-binary", "cox2"]:
        data_source = DiskDataSource(name)
        train, test, _ = data_source.dataset
        i = 11
        neighs = [sample_neigh(train, i) for j in range(10000)]
        clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
            nodes in neighs]
        path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
            for graph, nodes in neighs]
        #plt.subplot(1, 2, i-9)
        plt.scatter(clustering, path_length, s=10, label=name)
    plt.legend()
    plt.savefig("plots/clustering-vs-path-length.png")


# from common import models

"""Defines all graph embedding models"""



# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        if len(FEATURE_AUGMENT) > 0:
            self.feat_preprocess = Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
            args.conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3*hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        if args.conv_type == "PNA":
            post_input_dim *= 3
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            #return lambda i, h: pyg_nn.GINConv(nn.Sequential(
            #    nn.Linear(i, h), nn.ReLU()))
            return lambda i, h: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)
                ))
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return SAGEConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        #if data.x is None:
        #    data.x = torch.ones((data.num_nodes, 1), device=get_device())

        #x = self.pre_mp(x)
        if self.feat_preprocess is not None:
            if not hasattr(data, "preprocessed"):
                data = self.feat_preprocess(data)
                data.preprocessed = True
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type=="PNA" else
            len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                        self.convs_mean[i](curr_emb, edge_index),
                        self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                        self.convs_mean[i](emb, edge_index),
                        self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)
        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        #emb = self.batch_norm(emb)   # TODO: test
        #out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)




# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for order emb.
        The e term is the amount of violation (if b is a subgraph of a).
        For positive examples, the e term is minimized (close to 0); 
        for negative examples, the e term is trained to be at least greater than self.margin.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=get_device()), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
            device=get_device()), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss


class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
            out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        #edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        #return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)
        #aggr_out = torch.matmul(aggr_out, self.weight)

        #if self.bias is not None:
        #    aggr_out = aggr_out + self.bias

        #if self.normalize:
        #    aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# pytorch geom GINConv + weighted edges
class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
            edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
            edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


# from common import utils



def sample_neigh(graphs, size):
    ps = np.array([len(g) for g in graphs], dtype=float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh

cached_masks = None
def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    #v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    #v = [np.sum(v) for mask in cached_masks]
    return v

def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=np.int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=np.int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))

def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
            progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size: total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0/(k+1)) ** 1.5
    #ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts

def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
        node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
            not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
            else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            #if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
            node_anchored)
        sg.remove(w)

def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
    node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    #sizes = {}
    #for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        #bads, t = 0, 0
        #for ka, nas in counts.items():
        #    for kb, nbs in counts.items():
        #        if ka != kb:
        #            for a in nas:
        #                for b in nbs:
        #                    if nx.is_isomorphic(a, b):
        #                        bads += 1
        #                        print("bad", bads, t)
        #                    t += 1

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
            reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return device_cache

def parse_optimizer(parser):
    print ("Parsing optimizer")
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str, default="adam",
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str, default="none",
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int, default=0,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int, default=0,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float, default=0.0,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float, default=1.0,
            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float, default=0.0,
            help='Optimizer weight decay.')

    print ("Done parsing optimiser") 


def build_optimizer(args, params):
    print ("Building optimizer")
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    
    print ("Returning scheduler and optimizer")
    return scheduler, optimizer

def batch_nx_graphs(graphs, anchors=None):
    #motifs_batch = [pyg_from_networkx(
    #    nx.convert_node_labels_to_integers(graph)) for graph in graphs]
    #loader = DataLoader(motifs_batch, batch_size=len(motifs_batch))
    #for b in loader: batch = b
    augmenter = FeatureAugment()
    
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    batch = batch.to(get_device())
    return batch



# ++++++++++++++++++++++++++++++++++++++++++++++++++



# if HYPERPARAM_SEARCH:
#     from test_tube import HyperOptArgumentParser
#     from subgraph_matching.hyp_search import parse_encoder
# else:
    # from subgraph_matching.config import parse_encoder

import argparse

def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    #parse_optimizer(parser)
    print ("Parsing encoder")

    enc_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                        help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    enc_parser.add_argument('--skip', type=str,
                        help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                        help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                        help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                        help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--checkpoint_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
        help='tag to identify the run')


    enc_parser.add_argument('--try_alignment', action="store_true",
                        help='test our model`s retrieval capabilities')
    enc_parser.add_argument('--query_path', type=str, help='path of query graph',
        default="")
    enc_parser.add_argument('--target_path', type=str, help='path of target graph',
        default="")
    


    enc_parser.set_defaults(conv_type='SAGE',
                        method_type='order',
                        dataset='syn',
                        n_layers=8,
                        batch_size=64,
                        hidden_dim=64,
                        skip="learnable",
                        dropout=0.0,
                        n_batches=1000000,
                        opt='adam',   # opt_enc_parser
                        opt_scheduler='none',
                        opt_restart=100,
                        weight_decay=0.0,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=1000,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        tag='',
                        val_size=4096,
                        node_anchored=True)

    print ("Done parsing encoder")

    #return enc_parser.parse_args(arg_str)



# from subgraph_matching.test import validation

from collections import defaultdict
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score


USE_ORCA_FEATS = False # whether to use orca motif counts along with embeddings
MAX_MARGIN_SCORE = 1e9 # a very large margin score to given orca constraints

def validation(args, model, test_pts, logger, batch_n, epoch, verbose=False):
    # test on new motifs
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(get_device())
            pos_b = pos_b.to(get_device())
        neg_a = neg_a.to(get_device())
        neg_b = neg_b.to(get_device())
        labels = torch.tensor([1]*(pos_a.num_graphs if pos_a else 0) +
            [0]*neg_a.num_graphs).to(get_device())
        with torch.no_grad():
            emb_neg_a, emb_neg_b = (model.emb_model(neg_a),
                model.emb_model(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                    model.emb_model(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b
            pred = model(emb_as, emb_bs)
            raw_pred = model.predict(pred)
            if USE_ORCA_FEATS:
                import orca
                import matplotlib.pyplot as plt
                def make_feats(g):
                    counts5 = np.array(orca.orbit_counts("node", 5, g))
                    for v, n in zip(counts5, g.nodes):
                        if g.nodes[n]["node_feature"][0] > 0:
                            anchor_v = v
                            break
                    v5 = np.sum(counts5, axis=0)
                    return v5, anchor_v
                for i, (ga, gb) in enumerate(zip(neg_a.G, neg_b.G)):
                    (va, na), (vb, nb) = make_feats(ga), make_feats(gb)
                    if (va < vb).any() or (na < nb).any():
                        raw_pred[pos_a.num_graphs + i] = MAX_MARGIN_SCORE

            if args.method_type == "order":
                pred = model.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
                raw_pred *= -1
            elif args.method_type == "ensemble":
                pred = torch.stack([m.clf_model(
                    raw_pred.unsqueeze(1)).argmax(dim=-1) for m in model.models])
                for i in range(pred.shape[1]):
                    print(pred[:,i])
                pred = torch.min(pred, dim=0)[0]
                raw_pred *= -1
            elif args.method_type == "mlp":
                raw_pred = raw_pred[:,1]
                pred = pred.argmax(dim=-1)
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
        torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
        torch.sum(labels).item() if torch.sum(labels) > 0 else
        float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    auroc = roc_auc_score(labels, raw_pred)
    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    if verbose:
        import matplotlib.pyplot as plt
        precs, recalls, threshs = precision_recall_curve(labels, raw_pred)
        plt.plot(recalls, precs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("plots/precision-recall-curve.png")
        print("Saved PR curve plot in plots/precision-recall-curve.png")

    print("\n{}".format(str(datetime.now())))
    print("Validation. Epoch {}. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch,
            acc, prec, recall, auroc, avg_prec,
            tn, fp, fn, tp))
    wandb.log({"Validation Epoch": epoch, 
                "Validation accuracy": acc,
                "Validation precision": prec,
                "Validation recall": recall,
                "Validation average precision": avg_prec})        

    if not args.test:
        logger.add_scalar("Accuracy/test", acc, batch_n)
        logger.add_scalar("Precision/test", prec, batch_n)
        logger.add_scalar("Recall/test", recall, batch_n)
        logger.add_scalar("AUROC/test", auroc, batch_n)
        logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        logger.add_scalar("TP/test", tp, batch_n)
        logger.add_scalar("TN/test", tn, batch_n)
        logger.add_scalar("FP/test", fp, batch_n)
        logger.add_scalar("FN/test", fn, batch_n)
    
    print("Saving {}".format(args.model_path))
    torch.save(model.state_dict(), args.model_path)

    if verbose:
        conf_mat_examples = defaultdict(list)
        idx = 0
        for pos_a, pos_b, neg_a, neg_b in test_pts:
            if pos_a:
                pos_a = pos_a.to(get_device())
                pos_b = pos_b.to(get_device())
            neg_a = neg_a.to(get_device())
            neg_b = neg_b.to(get_device())
            for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
                if not list_a: continue
                for a, b in zip(list_a.G, list_b.G):
                    correct = pred[idx] == labels[idx]
                    conf_mat_examples[correct, pred[idx]].append((a, b))
                    idx += 1



def build_model(args):
    # build model
    print (f"We're using model of type {args.method_type}")
    if args.method_type == "order":
        model = OrderEmbedder(1, args.hidden_dim, args)
    elif args.method_type == "mlp":
        model = BaselineMLP(1, args.hidden_dim, args)
    model.to(get_device())
    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            print (f"Loading model from {args.checkpoint_path}...")
            model.load_state_dict(torch.load(args.checkpoint_path,
                map_location=get_device()))
    return model

def make_data_source(args):
    toks = args.dataset.split("-")
    if toks[0] == "syn":
        if len(toks) == 1 or toks[1] == "balanced":
            data_source = OTFSynDataSource(
                node_anchored=args.node_anchored)
        elif toks[1] == "imbalanced":
            data_source = OTFSynImbalancedDataSource(
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    else:
        if len(toks) == 1 or toks[1] == "balanced":
            data_source = DiskDataSource(toks[0],
                node_anchored=args.node_anchored)
        elif toks[1] == "imbalanced":
            data_source = DiskImbalancedDataSource(toks[0],
                node_anchored=args.node_anchored)
        else:
            raise Exception("Error: unrecognized dataset")
    return data_source

def train(args, model, logger, in_queue, out_queue):
    """Train the order embedding model.

    args: Commandline arguments
    logger: logger for logging progress
    in_queue: input queue to an intersection computation worker
    out_queue: output queue to an intersection computation worker
    """
    print ("Entering the main train function...")
    scheduler, opt = build_optimizer(args, model.parameters())
    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)

    done = False
    while not done:
        data_source = make_data_source(args)
        loaders = data_source.gen_data_loaders(args.eval_interval *
            args.batch_size, args.batch_size, train=True)
        for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
            msg, _ = in_queue.get()
            if msg == "done":
                done = True
                break
            # train
            print ("Here goes model.train()...")
            model.train()
            model.zero_grad()
            pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
                batch_neg_target, batch_neg_query, True)
            emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
            emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)
            #print(emb_pos_a.shape, emb_neg_a.shape, emb_neg_b.shape)
            emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
            emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            labels = torch.tensor([1]*pos_a.num_graphs + [0]*neg_a.num_graphs).to(
                get_device())
            intersect_embs = None
            print ("Generating model predictions...")
            pred = model(emb_as, emb_bs)
            print ("Calculating loss...")
            loss = model.criterion(pred, intersect_embs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler:
                scheduler.step()

            print ("OK, some more stuff is going on...")
            if args.method_type == "order":
                with torch.no_grad():
                    pred = model.predict(pred)
                model.clf_model.zero_grad()
                pred = model.clf_model(pred.unsqueeze(1))
                criterion = nn.NLLLoss()
                clf_loss = criterion(pred, labels)
                clf_loss.backward()
                clf_opt.step()
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            train_loss = loss.item()
            train_acc = acc.item()

            out_queue.put(("step", (loss.item(), acc)))

def train_loop(args):
    print ("Starting our train loop")
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    

    print("Starting {} workers".format(args.n_workers))
    in_queue, out_queue = mp.Queue(), mp.Queue()

    print("Using dataset {}".format(args.dataset))

    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)

    print ("Building model")
    model = build_model(args)
    model.share_memory()

    if args.method_type == "order":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)
    else:
        clf_opt = None

    print ("Making data source")
    data_source = make_data_source(args)
    loaders = data_source.gen_data_loaders(args.val_size, args.batch_size,
        train=False, use_distributed_sampling=False)
    test_pts = []
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))

    
    workers = []
    print ("starting workers")
    for i in range(args.n_workers):
        worker = mp.Process(target=train, args=(args, model, data_source,
            in_queue, out_queue))
        worker.start()
        workers.append(worker)

    if args.test:
        validation(args, model, test_pts, logger, 0, 0, verbose=True)
    else:
        batch_n = 0
        for epoch in trange(args.n_batches // args.eval_interval):
            for i in trange(args.eval_interval):
                in_queue.put(("step", None))
            for i in trange(args.eval_interval):
                msg, params = out_queue.get()
                train_loss, train_acc = params
                print("Batch {}. Loss: {:.4f}. Training acc: {:.4f}".format(
                    batch_n, train_loss, train_acc), end="               \r")
                wandb.log({"Batch": batch_n, "Train Loss": train_loss, "Train Acc": train_acc})
                logger.add_scalar("Loss/train", train_loss, batch_n)
                logger.add_scalar("Accuracy/train", train_acc, batch_n)
                batch_n += 1
            validation(args, model, test_pts, logger, batch_n, epoch, verbose=True)

    for i in trange(args.n_workers):
        in_queue.put(("done", None))
    for worker in workers:
        worker.join()



def gen_alignment_matrix(model, query, target, method_type="order"):
    """Generate subgraph matching alignment matrix for a given query and
    target graph. Each entry (u, v) of the matrix contains the confidence score
    the model gives for the query graph, anchored at u, being a subgraph of the
    target graph, anchored at v.

    Args:
        model: the subgraph matching model. Must have been trained with
            node anchored setting (--node_anchored, default)
        query: the query graph (networkx Graph)
        target: the target graph (networkx Graph)
        method_type: the method used for the model.
            "order" for order embedding or "mlp" for MLP model
    """

    mat = np.zeros((len(query), len(target)))
    for i, u in enumerate(query.nodes):
        for j, v in enumerate(target.nodes):
            batch = batch_nx_graphs([query, target], anchors=[u, v])
            embs = model.emb_model(batch)
            pred = model(embs[1].unsqueeze(0), embs[0].unsqueeze(0))
            raw_pred = model.predict(pred)
            if method_type == "order":
                raw_pred = torch.log(raw_pred)
            elif method_type == "mlp":
                raw_pred = raw_pred[0][1]
            mat[i][j] = raw_pred.item()
    return mat


def align(args, model):
    """Build an alignment matrix for matching a query subgraph in a target graph.
    Subgraph matching model needs to have been trained with the node-anchored option
    (default)."""
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")
    
    if args.query_path:
        with open(args.query_path, "rb") as f:
            query = pickle.load(f)
    else:
        query = nx.gnp_random_graph(8, 0.25)
    if args.target_path:
        with open(args.target_path, "rb") as f:
            target = pickle.load(f)
    else:
        target = nx.gnp_random_graph(16, 0.25)

       


    mat = gen_alignment_matrix(model, query, target,
        method_type=args.method_type)

    np.save("results/alignment.npy", mat)
    print("Saved alignment matrix in results/alignment.npy")

    plt.imshow(mat, interpolation="nearest")
    plt.savefig("plots/alignment.png")
    print("Saved alignment matrix plot in plots/alignment.png")


    print ("Here's our target graph:")
    plt.figure(1)
    nx.draw(target, with_labels=True)
    print ("And here's our query graph:")
    plt.figure(2)
    nx.draw(query, with_labels=True)
    print ("Before plt.show in align")
    plt.show()
    print ("After plt.show in align")


def try_alignment(args, n=3):
    for i in trange(n):
        print (f"Testing our model on a random query and target graph. Attempt #{i}")
        align(args, build_model(args))

    input ("Are you satisfied?")    



def main(force_test=False):
    print ("Spawning multiple Processes...")
    mp.set_start_method("spawn", force=True)
    print ("Constructing a parser...")
    parser = (argparse.ArgumentParser(description='Order embedding arguments')
        if not HYPERPARAM_SEARCH else
        HyperOptArgumentParser(strategy='grid_search'))

    parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True

    print ("Checking whether to try alignment...")
    if args.try_alignment:
        try_alignment(args, n=3)    

    # Currently due to parallelism in multi-gpu training, this code performs
    # sequential hyperparameter tuning.
    # All gpus are used for every run of training in hyperparameter search.
    if HYPERPARAM_SEARCH:
        for i, hparam_trial in enumerate(args.trials(HYPERPARAM_SEARCH_N_TRIALS)):
            print("Running hyperparameter search trial", i)
            print(hparam_trial)
            train_loop(hparam_trial)
    else:
        print ("Trying to init wandb logging for our multiprocessing training...") 
        wandb.init(project="neuromatch_experiments",
                group="MultiProcessing training")

        print ("We should proceed to training loop...")
        train_loop(args)

if __name__ == '__main__':
    print ("Hooray! Calling main")
    # datasets_main()
    # freeze_support()
    main()
