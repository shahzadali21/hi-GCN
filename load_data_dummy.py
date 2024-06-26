import networkx as nx
import numpy as np
import scipy.sparse as sp
import os

def read_graphfile(datadir, dataname, max_nodes=None):
    # Load dummy data
    adj = sp.load_npz(os.path.join(datadir, 'dummy_adj.npz'))
    features = sp.load_npz(os.path.join(datadir, 'dummy_features.npz')).toarray()
    labels = np.load(os.path.join(datadir, 'dummy_labels.npy'))

    G = nx.from_scipy_sparse_array(adj)
    for i, feat in enumerate(features):
        G.nodes[i]['feat'] = feat
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    return [G]
