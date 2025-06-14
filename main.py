import pandas as pd
import numpy as np
from graph import *
import random

# Lê o arquivo CSV
#df = pd.read_csv('netflix_amazon_disney_titles.csv', usecols=['title', 'director', 'cast'])
#df = df.head(100)
#print(df.head())

# Cria uma instância do grafo direcionado
graph_d = Graph_directed()
#graph_u = Graph_undirected()

graph_d = read_graph_csv('teste.csv', graph_d)

#Questão 1:
# Constrói o grafo com os dados
#graph_d, graph_u = construct_graph(graph_d, graph_u, df)

#Após a construção de cada grafo, retorne a quantidade de vértices e arestas.
graph_d.return_vertex_edges()
#graph_u.return_vertex_edges()

print(graph_d)