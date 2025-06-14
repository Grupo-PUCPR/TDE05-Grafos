import pandas as pd
import numpy as np
from graph import *
import random

graph_d = Graph_directed()
graph_u = Graph_undirected()
graph_d = read_graph_csv('teste.csv', graph_d)
print(graph_d)

graph_d.return_vertex_edges()
print(graph_d.dfs_kosarajus())

graph_d.kosarajus()
