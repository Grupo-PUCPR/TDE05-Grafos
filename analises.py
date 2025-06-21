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


#Questão 2
print("\n--- Questão 2 ---")
graph_d.kosarajus()
graph_u.return_components()


print("\n--- Questão 3 ---")
node = random.choice(list(graph_u.body))
mst, cost = graph_u.minimum_spannig_tree(node)
print(f"O nó esclhido foi: {node}, o custo da sua árvore mínima foi de {cost}, com os respectivos nós: \n{mst}")


print("\n--- Questão 4 ---")
graph_d.analyze_degree_centrality()
graph_u.analyze_degree_centrality()

print("\n--- Questão 5 ---")



print("\n--- Questão 6 ---")
graph_d.analyze_closeness_centrality()
graph_u.analyze_closeness_centrality()
