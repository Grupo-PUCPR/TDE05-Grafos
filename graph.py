import numpy as np
import pandas as pd
from collections import defaultdict
import os
import heapq

class Graph:
  def __init__(self):
      self.order = 0
      self.size = 0
      self.vertices = []
      self.body = defaultdict(dict) 

  def __str__(self):
    return self.print_list()
    
  def get_order(self):
    print(f"\nA ordem do grafo é: {self.order}")
    return self.order

  def get_size(self):
    print(f"\nO tamanho do grafo é: {self.size}")
    return self.size
  
  def return_vertex_edges(self):
    print(f"{self.__class__.__name__}:\nVértices: {self.order}\nArestas: {self.size}")
    return 
  
  def return_components(self):
    raise NotImplementedError("Tem que ser implementado na subclasse!")

  def add_vertex(self, name):
      if name not in self.vertices:
          self.vertices.append(name)
          self.order += 1
          print(name)
      else:
          raise ValueError("Vértice já existe!")

  def print_list(self):
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def add_edge(self, vertex1, vertex2, weight):
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def remove_edge(self, vertex1, vertex2):
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def print_graph(self):
    raise NotImplementedError("Tem que ser implementado na subclasse!")

  def remove_vertex(self, vertex):
      if vertex in self.vertices:
          # Remove todas as arestas conectadas
          for v in self.vertices:  # Copia para iterar de forma segura
            if self.has_edge(vertex, v):
              self.remove_edge(vertex, v)
            self.remove_edge(v, vertex)
          self.vertices.remove(vertex)
          del self.body[vertex]
          self.order -= 1
      else:
          raise ValueError("Vértice não existe!")

  def has_edge(self, vertex1, vertex2):
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def indegree(self, vertex):
      raise NotImplementedError("Tem que ser implementado na subclasse!")
    
  def outdegree(self, vertex):
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def degree(self, vertex):
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def get_weight(self, vertex1, vertex2):      
      raise NotImplementedError("Tem que ser implementado na subclasse!")

  def get_adjacent(self, node):
    if node not in self.body:
      raise ValueError("Este nó não existe!")
    else:
      adjs = []
      for v in self.body[node]:
        adjs.append(v)   
      return adjs
    
  def transpose(self):
    g_transpose = self.__class__()
    for v in self.body:
      for v1, weight in self.body[v].items():
        g_transpose.add_edge(v1, v, weight)
    return g_transpose

  def dfs_iterative(self, source_node):
    visited = []
    stack = []

    stack.append(source_node)

    while len(stack) > 0:
      element = stack.pop()

      if element not in visited:
        visited.append(element)

        for (adj,_) in self.body[element]:
          if adj not in visited:
            stack.append(adj)
    return visited

  def eulerian(self): 
    """
    Validates if the graph is Eulerian. First checks if the total degree is even; then if the indegree and outdegree
    of the vertex are equal; and finally, if the graph is connected.

    Returns:
        bool|str: Returns True if the graph is Eulerian, or returns error strings,
        informing what the problems of the graph are.:
        - "The total degree of a vertex is not even"
        - "There are one or more vertices with indegree different from outdegree"
        - "The graph is not connected"
    """

    invalidations = []
    # error message
    degree_in_diff_out = "There are one or more vertices with indegree different from outdegree"
    graph_is_weak = "The graph is not connected"

    eulerian_validation = True
    for vertex in self.body:
      if not(self.indegree(vertex) == self.outdegree(vertex)):
        invalidations.append(degree_in_diff_out) if degree_in_diff_out not in invalidations else None

    dfs = self.dfs_iterative(self.vertices[len(self.vertices) -1]) # checks if the graph is connected
    eulerian_validation = sorted(dfs) == sorted(self.vertices)

    invalidations.append(graph_is_weak) if not(eulerian_validation) else None

    error_message = ""
    for i, invalidation in enumerate(invalidations):
      error_message += (invalidation + ", " if i < len(invalidations) - 1 else invalidation)

    return eulerian_validation, error_message
  
  def diameter(self):
        largest_costs = []
        for node in self.vertices:
            lst = self.dijkstra(node)
            max_key, max_value = max(lst.items(), key=lambda item: item[1][0])

            max_path = [max_key]
            current_node = max_key
            while current_node != node:
                predecessor = lst[current_node][1]
                if predecessor is None:
                    break
                max_path.append(predecessor)
                current_node = predecessor

            max_path.reverse()
            largest_costs.append([max_value[0], max_path])

        return max(largest_costs, key=lambda item: item[0])
  
"""
Neste trabalho, você e sua equipe, irão explorar os relacionamentos entre criadores de conteúdo
disponível nas principais plataformas de streaming (Netflix, Amazon Prime Video e Disney+). A partir de
um conjunto de dados1 com informações sobre 19.621 filmes e séries envolvendo 61.811 atores/atrizes
e 10.870 diretores, você deverá construir dois grafos para a condução de análises exploratórias:

1. Um grafo ponderado direcionado que representa as relações entre os atores/atrizes com os
diretores das obras, considerando todos os filmes e séries do catálogo;

2. Um grafo ponderado não-direcionado que representa as relações entre os atores/atrizes em uma
obra, considerando todos os filmes e séries do catálogo.
"""

class Graph_directed(Graph):
  def __init__(self):
    super().__init__()
    self.body = defaultdict(dict)

  def add_edge(self, vertex1, vertex2, weight):
    if weight < 0:
      raise ValueError("Peso inválido!")
    
    # Adiciona vértices se não existirem
    if vertex1 not in self.vertices:
      self.add_vertex(vertex1)
    if vertex2 not in self.vertices:
      self.add_vertex(vertex2)

    self.body[vertex1][vertex2] = weight
    
    self.size += 1

  def remove_edge(self, vertex1, vertex2):
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      raise ValueError("Vértice não existe!")
    else:
      if vertex1 in self.body:
        self.body[vertex1].remove((vertex2))

  def get_weight(self, vertex1, vertex2):
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      return False
    else:
      if vertex1 in self.body:
        for v in self.body[vertex1]:
          if v[0] == vertex2:
            #print(f"{vertex1}: {v}")
            return v[1]
      else:
         raise ValueError("Não possuem arestas!")
  
  def dfs_visiting_finished(self, source_node):
    visited = []
    stack = []
    order_visited = []

    stack.append(source_node)

    while len(stack) > 0:
      element = stack.pop()
      
      if element not in visited:
        visited.append(element)

        for adj, _ in self.body(element):
          if adj not in visited:
            stack.append(adj)

    return visited
  

  def transpose_graph(self):
    graph_t = Graph_directed()
    for node in self.body:
      for adj_node, weight in self.body[node]:
        graph_t.add_edge(adj_node, node, weight) #add o inverso


class Graph_undirected(Graph):
  def __init__(self):
    super().__init__()
    self.body = defaultdict(dict)

  def add_edge(self, vertex1, vertex2, weight):
    if weight < 0:
      raise ValueError("Peso inválido!")
    
    # Adiciona vértices se não existirem
    if vertex1 not in self.vertices:
      self.add_vertex(vertex1)
    if vertex2 not in self.vertices:
      self.add_vertex(vertex2)

    self.body[vertex1][vertex2] = weight
    self.size += 1

  def return_components(self):
    pass

  def has_edge(self, vertex1, vertex2):
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      raise ValueError("Vértice não existe!")
    else:
      if vertex1 in self.body:
        for v in self.body[vertex1]:
          if v[0] == vertex2:
            return True
      else:
        return False
        
  def get_weight(self, vertex1, vertex2):
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      return False
    else:
      if vertex1 in self.body:
        for v in self.body[vertex1]:
          if v == vertex2:
            return self.body[vertex1][vertex2]
      else:
        return False

  def return_edge(self, vertex1):
    if vertex1 not in self.vertices:
      raise ValueError("Vértice não existe!")
    else:
      return self.body[vertex1]

def return_values(list_values):
    list_values = [d.strip() for d in list_values.split(',')]
    list_values = list(dict.fromkeys(list_values))
    return list_values

def work_together(actor, cast):
    return set(cast) - {actor}


#demora pq ele itera por tudo
def construct_graph(graph_d, graph_u, df):
  df = df.dropna() #já tiro todas as linhas que não tiverem um valor (NaN)
  for title, directors, cast in df.values:
    directors = str(directors)
    cast = str(cast)
    title = str(title)
  
    #Primeiro a construção do grafo direcionado
    directors = return_values(directors)
    for director in directors:
      director = format(director)
      if director not in graph_d.vertices:
        graph_d.add_vertex(director)
    
      #add cada um dos vértices de atores
    cast = return_values(cast)
    for actor in cast:
      actor = format(actor)
      if actor not in graph_d.vertices:
        graph_d.add_vertex(actor)
        graph_u.add_vertex(actor)

    for actor in cast:
      actor = format(actor)
      work_together_actor = work_together(actor, cast)
      for a in work_together_actor:
        a = format(a)
        weight = graph_u.get_weight(actor, a)
        if weight:
          graph_u.add_edge(actor, a, weight + 1)  # Adiciona com peso incrementado
        else:
          graph_u.add_edge(actor, a, 1)  # Primeira colaboração

    #add as arestas ponderadas
    for director in directors:
      for actor in cast:
        actor = format(actor)
        director = format(director)
        weight = graph_u.get_weight(actor, a)
        if weight:
          graph_d.add_edge(actor, director, weight + 1)  # Adiciona com peso incrementado
        else:
          graph_d.add_edge(actor, director, 1)  # Primeira colaboração

  return graph_d, graph_u

def save_graph_csv(graph, transpose=False):
  data = []
  for origem, destinos in graph.body.items():
      for destino, peso in destinos.items():
          data.append((origem, destino, peso))

  df = pd.DataFrame(data, columns=['Origem', 'Destino', 'Peso'])
  if transpose:
      df.to_csv(f'graph_{graph.__class__.__name__}_transpose.csv', index=False)
  else:
    df.to_csv(f'graph_{graph.__class__.__name__}.csv', index=False)

def format(name):
  name = name.split(" ")
  final_name = ''
  for n in name:
    final_name += n.capitalize()
  return final_name.replace(' ', "")#garante que tira os espaços