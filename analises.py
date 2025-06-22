import pandas as pd
import numpy as np
from graph import *

# Lê o arquivo CSV
df = pd.read_csv('netflix_amazon_disney_titles.csv', usecols=['title', 'director', 'cast'])
df = df.head(100)
print(df.head())

graph_teste_d = Graph_directed()
graph_teste_u = Graph_undirected()


# Cria uma instância do grafo direcionado
graph_d = Graph_directed()
graph_u = Graph_undirected()


#Questão 1:
graph_d, graph_u = construct_graph(graph_d, graph_u, df)

graph_d.return_vertex_edges()
graph_u.return_vertex_edges()

save_graph_csv(graph_d)
save_graph_csv(graph_u)







print("\n--- Questão 4 ---")
graph_d.analyze_degree_centrality()
graph_u.analyze_degree_centrality()

