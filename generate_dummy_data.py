import numpy as np
import scipy.sparse as sp
import os

# Create directory for dummy data (if it doesn't exist)
if not os.path.exists('data'):
    os.makedirs('data')

# Parameters as per README file
num_nodes = 100   # N: Number of nodes
num_features = 10  # D: Number of features per node
num_classes = 2   # E: Number of classes

# Adjacency matrix (N x N)
# Feature matrix (N x D)
# Label matrix (N x E)

# Generate dummy adjacency matrix
adj_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
np.fill_diagonal(adj_matrix, 0)  # No self-loops
adj_matrix = sp.coo_matrix(adj_matrix)

# Generate dummy feature matrix
features_matrix = np.random.rand(num_nodes, num_features)
features_matrix = sp.coo_matrix(features_matrix)

# Generate dummy label matrix
labels_matrix = np.eye(num_classes)[np.random.choice(num_classes, num_nodes)]

# Save matrices
sp.save_npz('data/dummy_adj.npz', adj_matrix)
sp.save_npz('data/dummy_features.npz', features_matrix)
np.save('data/dummy_labels.npy', labels_matrix)

print("Dummy data generated and saved successfully.")
