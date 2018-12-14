
from itertools import chain

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import networkx
from tqdm import tqdm


def read_matrices():
	"""Function returns adjacency matrices

	"""
	matrices = []
	for fname in ['0.edges', '107.edges', '348.edges']:
	    with open(f'./facebook/{fname}', 'r') as f:
	        fb_edges = [list(map(int, l.strip().split())) for l in f]

	    n_edges = max(chain(*fb_edges)) + 1
	    fb_matrix = np.zeros([n_edges, n_edges], dtype=bool)
	    for e in fb_edges:
	        fb_matrix[e[0], e[1]] = True
	        fb_matrix[e[1], e[0]] = True
	    matrices.append(fb_matrix)
	return matrices


def plot_graph(matrices, perplexity=30, n_iter=1000):
	"""Plots graph with random points and tsne embedded points

	Color of points means number of neighbors.

	"""
	fig, axes = plt.subplots(len(matrices), 2, figsize=[16, len(matrices) * 10])
	for i, matrix in tqdm(enumerate(matrices), desc='plotting', total=len(matrices)):
		matrix = matrix.copy()
		matrix[np.arange(len(matrix)), np.arange(len(matrix))] = 1

		xs = np.arange(len(matrix)).reshape(-1, 1, 1).repeat(len(matrix), axis=1)
		ys = np.arange(len(matrix)).reshape(1, -1, 1).repeat(len(matrix), axis=0)
		indices = np.concatenate([xs, ys], axis=-1)
		edges = indices[matrix > 0]
		edge_indices = [tuple([*e, dict()]) for e in edges]

		np.random.seed(775)
		random_pos = np.random.uniform(size=[len(matrix), 2])
		tsne_pos = TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(matrix)

		edges = list(map(list, edge_indices))
		edges = [tuple([*l[:2], dict()]) for l in edges]
		G = networkx.Graph()
		G.add_nodes_from(list(range(len(matrix))))
		G.add_edges_from(edges)

		axes[i][0].set_title('random plot')
		axes[i][1].set_title('tsne plot')
		colors = np.sum(matrix, axis=1)
		vmin = np.min(colors)
		vmax = np.max(colors)
		cmap = plt.cm.coolwarm
		common_args=dict(with_labels=False, node_size=30, width=0.3,
		                 node_color=colors, edge_color='grey',
		                 vmin=vmin, vmax=vmax, cmap=cmap)

		networkx.drawing.draw_networkx(G, pos=random_pos, ax=axes[i][0], **common_args)
		networkx.drawing.draw_networkx(G, pos=tsne_pos, ax=axes[i][1], **common_args)
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
		sm.set_array([])
		cbar = plt.colorbar(sm, ax=axes[i][0])
	plt.savefig('./graphs_plot.jpg')


if __name__ == '__main__':
	matrices = read_matrices()
	plot_graph(matrices)
