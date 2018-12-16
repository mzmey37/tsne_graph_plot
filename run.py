
from itertools import chain
import argparse

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx
from tqdm import tqdm


def read_matrices(filenames):
	"""Function returns adjacency matrices

	"""
	matrices = []
	for fname in filenames:
		# read lines
	    with open(fname, 'r') as f:
	        edges = [list(map(int, l.strip().split())) for l in f]

	    # define mapping from nodes to their indices
	    node_set = set(chain(*edges))

	    n_nodes = len(node_set)
	    mapping = {n: i for i, n in enumerate(sorted(node_set))}

	    # create adjacency matrix
	    matrix = np.zeros([n_nodes, n_nodes], dtype=bool)
	    for e in edges:
	        matrix[mapping[e[0]], mapping[e[1]]] = True
	        matrix[mapping[e[1]], mapping[e[0]]] = True
	    matrices.append(matrix)

	return matrices


def read_other_coords(coord_paths):
	other_coords = []
	for coord_path in coord_paths:
		with open(coord_path, 'r', encoding='utf-8') as f:
			nodes = set()
			for l in f:
				node_id, *coords = l.strip().split('\t')
				nodes.add((node_id, *coords))
		other_coords.append(np.array([each[1:] for each in sorted(nodes, key=lambda x: x[0])], dtype=np.float32))
	return other_coords


def calculate_tsne_coords(matrices, perplexity, n_iter):
	tsne_matrices = []
	for matrix in tqdm(matrices, desc='calculating..', total=len(matrices)):
		matrix = matrix.copy()
		matrix[np.arange(len(matrix)), np.arange(len(matrix))] = 1
		tsne_pos = TSNE(2, perplexity=perplexity, n_iter=n_iter, metric='cosine').fit_transform(matrix)
		tsne_matrices.append(tsne_pos)
	return tsne_matrices


def calculate_random_coords(matrices):
	np.random.seed(777)
	return [np.random.uniform(size=[len(matr), 2]) for matr in matrices]


def make_graphs(matrices):
	graphs = []
	for matrix in matrices:
		graphs.append(networkx.from_numpy_matrix(matrix))
	return graphs


def plot_graphs(matrices, graphs, tsne_coords, other_coords, out_path, perplexity, n_iter):
	"""

	Color of points expresses number of neighbors.

	"""

	# prepare figure
	fig, axes = plt.subplots(len(matrices), 2, figsize=[30, len(matrices) * 10], squeeze=False)
	for ax, matrix, graph, tsne_matr, other_matr in tqdm(zip(axes, matrices, graphs, tsne_coords, other_coords),
		                                                 desc='plotting', total=len(tsne_coords)):
		# set labels and other metadata
		ax[0].set_title('other plot')
		ax[1].set_title('tsne plot')
		colors = np.sum(matrix, axis=1)
		vmin = np.min(colors)
		vmax = np.max(colors)
		cmap = plt.cm.coolwarm
		common_args=dict(with_labels=False, node_size=30, width=0.3,
		                 node_color=colors, edge_color='grey',
		                 vmin=vmin, vmax=vmax, cmap=cmap)

		# plot networkx
		networkx.drawing.draw_networkx(graph, pos=other_matr, ax=ax[0], **common_args)
		networkx.drawing.draw_networkx(graph, pos=tsne_matr, ax=ax[1], **common_args)

		# add colorbar
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
		sm.set_array([])
		cbar = plt.colorbar(sm, ax=ax[0], orientation="horizontal", pad=0.2)
		cbar = plt.colorbar(sm, ax=ax[1], orientation="horizontal", pad=0.2)
	plt.savefig(out_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('graph_paths', nargs='+', type=str, help="path to graph with nodes in format node_id\\tadjacent node_id")
	parser.add_argument('-out_path', type=str, default='./result.jpg', help='path to output of plot')
	parser.add_argument('-coord_paths', nargs='+', type=str, help='path to graph coordinates in format node_id\\tx1\\tx2[\\tx3]')
	parser.add_argument('-perplexity', type=float, default=15., help='perplexity')
	parser.add_argument('-n_iter', type=int, default=1000, help='number of iterations for tsne')

	args = parser.parse_args()
	print(args)
	matrices = read_matrices(args.graph_paths)
	tsne_coords = calculate_tsne_coords(matrices, args.perplexity, args.n_iter)
	if args.coord_paths is None:
		other_coords = calculate_random_coords(matrices)
	else:
		other_coords = read_other_coords(args.coord_paths)
		print(other_coords)

	graphs = make_graphs(matrices)
	plot_graphs(matrices, graphs, tsne_coords, other_coords, args.out_path, args.perplexity, args.n_iter)
