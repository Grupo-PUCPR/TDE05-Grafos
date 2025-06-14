import pandas as pd
import numpy as np
from graph import *
import random

"""1) (1 ponto) Construção dos dois grafos solicitados (direcionado e não-direcionado) utilizando lista de
adjacências. Durante o processo de construção, todos os nomes devem ser padronizados em letras
maiúsculas e sem espaços em branco no início e no final da string. Entradas do conjunto de dados
onde o nome do diretor e/ou nome do elenco estão vazias, devem ser ignoradas. Após a construção
de cada grafo, retorne a quantidade de vértices e arestas."""
df = pd.read_csv('netflix_amazon_disney_titles.csv', usecols=['title', 'director', 'cast'])
df = df.head(5000)

graph_d = Graph_directed()
graph_u = Graph_undirected()
graph_u = Graph_undirected()

graph_d, graph_u = construct_graph(graph_d, graph_u, df)

graph_d.return_vertex_edges()
graph_u.return_vertex_edges()

"""2) (1 ponto) Função para a identificação e contagem de componentes. Para o grafo direcionado, a função
deve contar a quantidade de componentes fortemente conexas. Para o grafo não-direcionado, a
função deve retornar a quantidade de componentes conexas."""
graph_d.kosarajus()
graph_u.return_components()

"""3) (1 ponto) Função que recebe como entrada um vértice X (por exemplo, BOB ODENKIRK) e retorna a
Árvore Geradora Mínima da componente que contêm o vértice X, bem como o custo total da árvore
(i.e., a soma dos pesos das arestas da árvore). Essa função deve ser executada somente no grafo não-
direcionado."""
node = random.choice(list(graph_u.body))
mst, cost = graph_u.minimum_spannig_tree(node)
print(node)
print(cost)
print(mst)

"""
4) (1 ponto) Função que calcula a Centralidade de Grau (Degree Centrality) de um vértice, retornando
um valor entre 0 e 1.
5) (1 ponto) Função que calcula a Centralidade de Intermediação (Betweenness Centrality) de um vértice,
retornando um valor entre 0 e 1.
6) (1 ponto) Função que calcula a Centralidade de Proximidade (Closeness Centrality) de um vértice,
retornando um valor entre 0 e 1."""

