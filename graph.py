import numpy as np
import pandas as pd
from collections import defaultdict
import os
import heapq

class Graph:
  def __init__(self):
      self.order = 0
      self.size = 0
      self.vertices = []  #Mantém a ordem de inserção
      self.body = defaultdict(list)  #Lista de adjacência com defaultdict

  def __str__(self):
    return self.print_list()
    
  def get_order(self):
    print(f"\nA ordem do grafo é: {self.order}")
    return self.order

  def get_size(self):
    print(f"\nO tamanho do grafo é: {self.size}")
    return self.size

  def add_vertex(self, name):
      if name not in self.vertices:
          self.vertices.append(name)
          self.order += 1
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

  def isolated_vertices(self):
    isolated = [v for v in self.vertices if self.degree(v) == 0]
    print(f"\n{len(isolated)} vértices isolados: {isolated}")

  def return_node(self, name):
    return self.body[name]

  def get_adjacent(self, node):
    if node not in self.body:
      raise ValueError("Este nó não existe!")
    else:
      adjs = []
      for v in self.body[node]:
        adjs.append(v)   
      return adjs

  def dijkstra(self, source_node):
    #create a dict with the structure: Vertex - Accumulated Weight, Predecessor
    distance = {vertex: [np.inf, None] for vertex in self.body}
    distance[source_node][0] = 0 
    cost = [(0, source_node)] #add the source node to the cost
    
    #while my cost list is not 0, continue
    while cost:
      accumulated_weight, current_node = heapq.heappop(cost)
      #get the adjacent vertices to the current node
      adjacents = self.get_adjacent(current_node)
      for vertex, edge_weight in adjacents:
          weight = accumulated_weight + edge_weight
          if weight < distance[vertex][0]:
            distance[vertex] = weight, current_node
            heapq.heappush(cost, (weight, vertex)) #add all adjacent vertices and their weight to the cost list
    
    #remove from the distance list the vertices whose values are inf
    unreachable_nodes = [v for v in distance if distance[v][0] == np.inf]
    for v in unreachable_nodes:
      distance.pop(v)

    return distance
  
  def list_distances(self, distance, node):
    lst = self.dijkstra(node)
    lst = [key for key, value in lst.items() if value[0] <= distance]
    print(f'\nOs vértices estão a uma distância abaixo de {distance}: {lst}')

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
  
class Graph_directed(Graph):
  def __init__(self):
    super().__init__()
    self.body = defaultdict(list)

  def add_edge(self, vertex1, vertex2, weight):
    if weight < 0:
      raise ValueError("Peso inválido!")
    
    # Adiciona vértices se não existirem
    if vertex1 not in self.vertices:
      self.add_vertex(vertex1)
    if vertex2 not in self.vertices:
      self.add_vertex(vertex2)

    for i in range(len(self.body[vertex1])):
      if self.body[vertex1][i][0] == vertex2:
          self.body[vertex1][i][1] = weight
          return
    
    #só add se não existir
    self.body[vertex1].append([vertex2, weight])
    self.size += 1

  def remove_edge(self, vertex1, vertex2):
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      raise ValueError("Vértice não existe!")
    else:
      if vertex1 in self.body:
        self.body[vertex1].remove((vertex2))

  def get_weight(self, vertex1, vertex2):
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      raise ValueError("Vértice não existe!")
    else:
      if vertex1 in self.body:
        for v in self.body[vertex1]:
          if v[0] == vertex2:
            print(f"{vertex1}: {v}")
            return v[1]
      else:
         raise ValueError("Não possuem arestas!")

  def construct_graph(self, df):
    df = df.drop(columns=['title','show_id', 'type', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description'])
    for directors, cast in df.values:
      directors = str(directors)
      cast = str(cast)
      if directors == 'nan' or directors == np.nan:
        continue
      else:
          #add cada um dos vértices de diretores
          directors = return_values(directors)
          for director in directors:
            if director not in self.vertices:
              self.add_vertex(director)
      if cast == 'nan' or cast == np.nan:
        continue
      else:
        #add cada um dos vértices de atores
        cast = return_values(cast)
        for actor in cast:
          if actor not in self.vertices:
            self.add_vertex(actor)

        #add as arestas ponderadas
        if directors != 'nan' or directors != np.nan:
          for director in directors:
            for actor in cast:
              try:
                weight = self.get_weight(actor, director)
                self.add_edge(actor, director, weight + 1)  # Adiciona com peso incrementado
              except:
                self.add_edge(actor, director, 1)  # Primeira colaboração
    print(self.body)

"""Na construção do primeiro grafo (direcionado), é necessário estabelecer conexões ponderadas de acordo
com a quantidade de colaborações partindo de cada ator/atriz até o nome de cada diretor. Na figura
abaixo, o processo é ilustrado considerando os 10 atores e 1 dos diretores. No entanto, o processo deve
ser realizado para cada um dos diretores listados (Michelle MacLaren, Adam Bernstein, etc)."""

def return_values(list_values):
    list_values = [d.strip() for d in list_values.split(',')]
    list_values = list(dict.fromkeys(list_values))
    return list_values

    graph_dot = pydot.Dot(graph_type='digraph', rankdir='LR')  # Grafo direcionado, da esquerda pra direita

    # Adiciona nós
    for vertex in graph.vertices:
        safe_vertex = str(vertex).replace('"', '\\"')  # escapa aspas
        node = pydot.Node(f'"{safe_vertex}"')  # força aspas ao redor
        graph_dot.add_node(node)

    # Adiciona arestas com pesos
    for vertex in graph.vertices:
        for neighbor, weight in graph.body[vertex]:
            safe_vertex = str(vertex).replace('"', '\\"')
            safe_neighbor = str(neighbor).replace('"', '\\"')
            edge = pydot.Edge(f'"{safe_vertex}"', f'"{safe_neighbor}"', label=str(weight))
            graph_dot.add_edge(edge)

    # Salva o grafo
    graph_dot.write_png(filename)