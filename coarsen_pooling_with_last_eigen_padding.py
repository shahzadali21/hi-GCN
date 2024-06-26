import graph
import networkx 
import matplotlib.pyplot as plt
from networkx.algorithms import community

import numpy as np
from sklearn.cluster import SpectralClustering
from scipy import sparse as sp
import scipy
import torch
from utils import sparse_mx_to_torch_sparse_tensor


def adj2edgeindex(adj):
    # Convert a sparse adjacency matrix to a PyTorch edge index format.
	adj = adj.tocoo().astype(np.float32)
	row = adj.row
	col = adj.col

	edge_index = torch.LongTensor([list(row),list(col)])

	return edge_index




class Graphs():
    # Graph class that supports coarsening pooling with spectral clustering.
	def __init__(self, adjacency_matrix, pooling_sizes):
		self.adjacency_matrix = adjacency_matrix
		self.num_nodes = adjacency_matrix[:,0].shape[0]
		self.pooling_sizes = pooling_sizes
		self.graphs = [scipy.sparse.csr_matrix(adjacency_matrix)]
		self.layer2pooling_matrices=dict()



	def coarsening_pooling(self, normalize = True):
    	# Apply spectral clustering-based coarsening and pooling to the graph.
		adj = scipy.sparse.csr_matrix(self.adjacency_matrix)
		for i in range(len(self.pooling_sizes)):
			adj_coarsened, pooling_matrices = self._coarserning_pooling_(adj, self.pooling_sizes[i],normalize)
			pooling_matrices = np.array(pooling_matrices)

			self.graphs.append(adj_coarsened)
			self.layer2pooling_matrices[i] = pooling_matrices
			adj = scipy.sparse.csr_matrix(adj_coarsened)


		#num_nodes_before_final = adj_coarsened.shape[0]
		#if num_nodes_before_final < 4:
			#num_nodes_before_final = 4
   
		num_nodes_before_final = 4		
		pooling_matrices_final = [sp.lil_matrix((adj_coarsened.shape[0],1)) for i in range(num_nodes_before_final)]
  			
		if adj_coarsened.shape[0]>1:
      		# Compute the Laplacian and its Fourier basis
			L_i = graph.laplacian(adj_coarsened, normalize)
			lamb_i, U_i = graph.fourier(L_i)

			for j in range(num_nodes_before_final):
				if j < adj_coarsened.shape[0]:
					if U_i[0,j] < 0:
						pooling_matrices_final[j][:,0] = -U_i[:,j].reshape(-1,1)
					else:
						pooling_matrices_final[j][:,0] = U_i[:,j].reshape(-1,1)
				else:
					if U_i[0, adj_coarsened.shape[0]-1] < 0:
						pooling_matrices_final[j][:,0] = -U_i[:, adj_coarsened.shape[0]-1].reshape(-1,1)
					else:
						pooling_matrices_final[j][:,0] = U_i[:, adj_coarsened.shape[0]-1].reshape(-1,1)

		else:
			for j in range(num_nodes_before_final):
				pooling_matrices_final[j][:,0] = adj_coarsened.reshape(-1,1)


		self.layer2pooling_matrices[i+1] = pooling_matrices_final

	def prepare_for_pytorch(self):
		# Prepare graphs and pooling matrices for PyTorch operations.
		self.edge_index_lists = [0]*len(self.graphs)
		for i in range(len(self.graphs)):
			self.edge_index_lists[i] = adj2edgeindex(self.graphs[i])
		for i in self.layer2pooling_matrices:
			self.layer2pooling_matrices[i] = [sparse_mx_to_torch_sparse_tensor(spmat).t() for spmat in self.layer2pooling_matrices[i]]




	def _coarserning_pooling_(self, adjacency_matrix, pooling_size, normalize=False):
		num_nodes = adjacency_matrix[:,0].shape[0]
		A_dense = adjacency_matrix.todense()
		num_clusters = int(num_nodes/pooling_size)

		# Apply spectral clustering to divide nodes into clusters
		if num_clusters == 0:
			num_clusters = num_clusters + 1
		sc = SpectralClustering(n_clusters = num_clusters, affinity= 'precomputed', n_init=10)
		sc.fit(A_dense)

		# Create a mapping from cluster labels to nodes
		clusters = dict()
		for inx, label in enumerate(sc.labels_):
			if label not in clusters:
				clusters[label] = []
			clusters[label].append(inx)
		num_clusters = len(clusters)

		# Determine the maximum size of clusters for pooling matrices
		num_nodes_in_largest_clusters = 0
		for label in clusters:
			if len(clusters[label])>=num_nodes_in_largest_clusters:

				num_nodes_in_largest_clusters = len(clusters[label])
		if num_nodes_in_largest_clusters <=5:
			num_nodes_in_largest_clusters = 5

		num_nodes_in_largest_clusters = 5 

		Adjacencies_per_cluster = [adjacency_matrix[clusters[label],:][:,clusters[label]] for label in range(len(clusters))]

		# Initialize an empty internal adjacency matrix
		A_int = sp.lil_matrix(adjacency_matrix)
		for i in range(len(clusters)):
			zero_list = list(set(range(num_nodes)) - set(clusters[i]))
			for j in clusters[i]:
				A_int[j,zero_list] = 0
				A_int[zero_list,j] = 0


######## Getting adjacenccy matrix wuith only external links
		A_ext = adjacency_matrix - A_int
######## Getting cluster vertex indicate matrix
		row_inds = []
		col_inds = []
		data = []

		for i in clusters:
			for j in clusters[i]:
				row_inds.append(j)
				col_inds.append(i)
				data.append(1)
		
		Omega = sp.coo_matrix((data,(row_inds,col_inds)))
		A_coarsened = np.dot( np.dot(np.transpose(Omega),A_ext), Omega)

########## Constructing pooling matrix
		pooling_matrices = [sp.lil_matrix((num_nodes,num_clusters)) for i in range(num_nodes_in_largest_clusters)]

		for i in clusters:
			adj = Adjacencies_per_cluster[i]

			# Compute the Laplacian and its Fourier basis for each cluster
			if len(clusters[i])>1:
				L_i = graph.laplacian(adj, normalize)
				lamb_i, U_i = graph.fourier(L_i)

				for j in range(num_nodes_in_largest_clusters):
					if j<len(clusters[i]): 
						if U_i[0,j] <0:
							pooling_matrices[j][clusters[i],i] = - U_i[:,j].reshape(-1,1)
						else:
							pooling_matrices[j][clusters[i],i] =  U_i[:,j].reshape(-1,1)
					else:
						if U_i[0, len(clusters[i])-1] <0:
							pooling_matrices[j][clusters[i],i] = - U_i[:, len(clusters[i])-1].reshape(-1,1)
						else:
							pooling_matrices[j][clusters[i],i] =  U_i[:, len(clusters[i])-1].reshape(-1,1)
			else:
				for j in range(num_nodes_in_largest_clusters):
					pooling_matrices[j][clusters[i],i] =  adj.reshape(-1,1)

		return A_coarsened, pooling_matrices
	