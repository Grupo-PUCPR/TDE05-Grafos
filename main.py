import pandas as pd
import numpy as np
from graph import *
import random
import pickle
import os
import time

directed_graph_file = 'graph_d.pkl'
undirected_graph_file = 'graph_u.pkl'

overall_start_time = time.time()

graph_d = None
graph_u = None


"""1) (1 ponto) Construção dos dois grafos solicitados (direcionado e não-direcionado) utilizando lista de
adjacências. Durante o processo de construção, todos os nomes devem ser padronizados em letras
maiúsculas e sem espaços em branco no início e no final da string. Entradas do conjunto de dados
onde o nome do diretor e/ou nome do elenco estão vazias, devem ser ignoradas. Após a construção
de cada grafo, retorne a quantidade de vértices e arestas."""
# df = pd.read_csv('netflix_amazon_disney_titles.csv', usecols=['title', 'director', 'cast'])
# df = df.head(100)

# serealizando para otimizar a abertura
if os.path.exists(directed_graph_file) and os.path.exists(undirected_graph_file):
    print("Carregando grafos a partir dos arquivos .pkl (versão rápida)...")
    load_start_time = time.time()
    
    with open(directed_graph_file, 'rb') as f:
        graph_d = pickle.load(f)
    
    with open(undirected_graph_file, 'rb') as f:
        graph_u = pickle.load(f)
        
    print("Grafos carregados com sucesso!")
    print(f"Tempo de carregamento dos grafos: {time.time() - load_start_time:.4f} segundos")

else:
    # Se os arquivos não existem, executa a construção completa
    print("Arquivos .pkl não encontrados. Construindo grafos a partir do CSV (primeira execução)...")
    build_start_time = time.time()

    # Instancia os grafos
    graph_d = Graph_directed()
    graph_u = Graph_undirected()
    
    # Lê o dataset
    df = pd.read_csv('netflix_amazon_disney_titles.csv', usecols=['title', 'director', 'cast'])
    
    # Constrói os grafos (processo demorado)
    graph_d, graph_u = construct_graph(graph_d, graph_u, df)
    print(f"Tempo de construção dos grafos: {time.time() - build_start_time:.4f} segundos")
    
    # --- SALVA OS GRAFOS PARA USO FUTURO ---
    print("Salvando grafos em arquivos .pkl para execuções futuras...")
    with open(directed_graph_file, 'wb') as f:
        pickle.dump(graph_d, f)
        
    with open(undirected_graph_file, 'wb') as f:
        pickle.dump(graph_u, f)
        
    print("Grafos salvos com sucesso!")

q1_start_time = time.time()

graph_d.return_vertex_edges()
graph_u.return_vertex_edges()
# print("directed")
# print(f"    {graph_d}")
# print("\n"*3)
# print("undirected")
# print(f"    {graph_u}")



# # save_graph_csv(graph_d)
# # save_graph_csv(graph_u)

print(f"Tempo da Questão 1: {time.time() - q1_start_time:.4f} segundos")

"""2) (1 ponto) Função para a identificação e contagem de componentes. Para o grafo direcionado, a função
deve contar a quantidade de componentes fortemente conexas. Para o grafo não-direcionado, a
função deve retornar a quantidade de componentes conexas."""
print("\n--- Questão 2 ---")
q2_start_time = time.time()
graph_d.kosarajus()
graph_u.return_components()
print(f"Tempo da Questão 2: {time.time() - q2_start_time:.4f} segundos")

"""3) (1 ponto) Função que recebe como entrada um vértice X (por exemplo, BOB ODENKIRK) e retorna a
Árvore Geradora Mínima da componente que contêm o vértice X, bem como o custo total da árvore
(i.e., a soma dos pesos das arestas da árvore). Essa função deve ser executada somente no grafo não-
direcionado."""
print("\n--- Questão 3 ---")
q3_start_time = time.time()
node = random.choice(list(graph_u.body))
mst, cost = graph_u.minimum_spannig_tree(node)
print(f"O nó esclhido foi: {node}, o custo da sua árvore mínima foi de {cost}, com os respectivos nós: \n{mst}")
print(f"Tempo da Questão 3: {time.time() - q3_start_time:.4f} segundos")

"""
4) (1 ponto) Função que calcula a Centralidade de Grau (Degree Centrality) de um vértice, retornando
um valor entre 0 e 1.
"""
print("\n--- Questão 4 ---")
q4_start_time = time.time()
graph_d.analyze_degree_centrality()
graph_u.analyze_degree_centrality()
print(f"Tempo da Questão 4: {time.time() - q4_start_time:.4f} segundos")

"""
5) (1 ponto) Função que calcula a Centralidade de Intermediação (Betweenness Centrality) de um vértice,
retornando um valor entre 0 e 1."""


"""
6) (1 ponto) Função que calcula a Centralidade de Proximidade (Closeness Centrality) de um vértice,
retornando um valor entre 0 e 1."""

print("\n--- Questão 6 ---")
overall_end_time = time.time()
graph_d.analyze_closeness_centrality()
graph_u.analyze_closeness_centrality()
print(f"\nTempo total de execução do script: {overall_end_time - overall_start_time:.4f} segundos")

