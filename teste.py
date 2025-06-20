import pandas as pd
import numpy as np
from graph import *
import random


graph_d = Graph_directed()
graph_u = Graph_undirected()

read_graph_csv('teste_mst.csv', graph_u)
read_graph_csv('teste_mst.csv', graph_d)


graph_d.return_vertex_edges()
graph_u.return_vertex_edges()

save_graph_csv(graph_d)
save_graph_csv(graph_u)

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
print(f"O nó esclhido foi: {node}, o custo da sua árvore mínima foi de {cost}, com os respectivos nós: \n{mst}")
