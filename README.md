# TSNE graph plotting tool
There are many ways how you can plot the graph, here is implementation of one of them.

## Installation

```
git clone https://github.com/mzmey37/tsne_graph_plot.git
cd tsne_graph_plot
pip install -r requirements.txt
```

## If you want too look how it works

```
./run_demo.sh
```
This will download example data and run algorithm. You will be able to see plot in ./result.jpg

## If you want to run it on your data and compare to your visualization technics


```
python run.py [-h] [-out_path OUT_PATH]
              [-coord_paths COORD_PATHS [COORD_PATHS ...]]
              [-perplexity PERPLEXITY] [-n_iter N_ITER]
              graph_paths [graph_paths ...]

positional arguments:
  graph_paths           space separated paths to graphs with nodes in format node_id\tadjacent
                        node_id\n...

optional arguments:
  -h, --help            show this help message and exit
  -out_path OUT_PATH    path to output of plot
  -coord_paths COORD_PATHS [COORD_PATHS ...]
                        path to your node coordinates in format
                        node_id\tx1\tx2\n...
  -perplexity PERPLEXITY
                        perplexity
  -n_iter N_ITER        number of iterations for tsne
```

