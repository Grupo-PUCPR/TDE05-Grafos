import pandas as pd
import numpy as np
from graph import *

# Lê o arquivo CSV
df = pd.read_csv('netflix_amazon_disney_titles.csv', usecols=['title', 'director', 'cast'])
df = df.head(100)
print(df.head())

# Cria uma instância do grafo direcionado
graph_d = Graph_directed()
graph_u = Graph_undirected()


#Questão 1:
# Constrói o grafo com os dados
graph_d, graph_u = construct_graph(graph_d, graph_u, df)

#Após a construção de cada grafo, retorne a quantidade de vértices e arestas.
graph_d.return_vertex_edges()
graph_u.return_vertex_edges()

save_graph_csv(graph_d)
save_graph_csv(graph_u)

gd_transpose = graph_d.transpose()
save_graph_csv(gd_transpose, True)