import numpy as np
import pandas as pd
from collections import defaultdict
import random

class Graph:
  def __init__(self):
      self.order = 0
      self.size = 0
      self.vertices = []
      self.body = defaultdict(dict) 

  def __str__(self):
    result = ""
    for origem, destinos in self.body.items():
        adjacentes = ', '.join([f'{destino}({peso})' for destino, peso in destinos.items()])
        result += f'{origem} -> {adjacentes}\n'
    return result
    
  def get_order(self):
    print(f"\nA ordem do grafo é: {self.order}")
    return self.order

  def get_size(self):
    print(f"\nO tamanho do grafo é: {self.size}")
    return self.size
  
  def return_vertex_edges(self):
    print(f"{self.__class__.__name__}:\nVértices: {self.order}\nArestas: {self.size}\n")
    return 
  
  def return_components(self):
    raise NotImplementedError("Tem que ser implementado na subclasse!")

  def add_vertex(self, name):
      if name.strip() not in self.vertices:
          self.vertices.append(name)
          self.order += 1
      else:
          raise ValueError("Vértice já existe!")    

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

  def dfs_iterative(self, source_node, global_visited):
    stack = [source_node]
    visited = []
    while len(stack) > 0:
      element = stack.pop()

      if element not in visited and element not in global_visited:
        visited.append(element)

        for (adj,_) in self.body[element].items():
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
  
  def dfs_kosarajus(self):
    visited = set()
    timestamps = {}
    count = 1
    nodes_to_visit = set(self.body)
    source_node = random.choice(list(nodes_to_visit))


    while nodes_to_visit:
        # Se não for a primeira rodada, pega qualquer nó que sobrou
        if source_node not in nodes_to_visit:
            source_node = random.choice(list(nodes_to_visit))

        stack = [(source_node, 'visit')]

        while stack:
            node, state = stack.pop()

            if state == 'visit':
                if node not in visited:
                    visited.add(node)
                    timestamps[node] = [count, None]  # tempo de entrada
                    count += 1

                    stack.append((node, 'post'))

                    for adj in self.body[node]:
                        if adj not in visited:
                            stack.append((adj, 'visit'))

            elif state == 'post':
                timestamps[node][1] = count  # tempo de saída
                count += 1

        nodes_to_visit -= visited

    return timestamps

  def transpose_graph(self):
    graph_t = Graph_directed()
    for node in self.body:
      for adj_node, weight in self.body[node].items():
        graph_t.add_edge(adj_node, node, weight) #add o inverso
    return graph_t
  
  def kosarajus(self):
    timestamps = self.dfs_kosarajus()
    graph_t = self.transpose_graph()

    timestamps = {n:t[1] for n, t in timestamps.items()}
    #pego somente o nome do ver, para cada um dos meus tempos, comparando e deixando em ordem do maior para o menor
    nodes = [n for n, t in sorted(timestamps.items(), key=lambda x: x[1], reverse=True)]

    scc = []
    visited_global = set()

    for node in nodes:
        if node not in visited_global:
            visited = graph_t.dfs_iterative(node, list(visited_global))  # lista de nós visitados na DFS
            scc.append(visited)
            visited_global.update(visited)

    print(f"Quantidade de componentes: {len(scc)}")


  def has_edge(self, vertex1, vertex2):
    """Verifica se existe uma aresta direcionada de vertex1 para vertex2"""
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      return False
    return vertex2 in self.body[vertex1]

  def indegree(self, vertex):
    """Calcula o grau de entrada de um vértice"""
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    count = 0
    for v in self.vertices:
      if vertex in self.body[v]:
        count += 1
    return count

  def outdegree(self, vertex):
    """Calcula o grau de saída de um vértice"""
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    return len(self.body[vertex])

  def degree(self, vertex):
    """Calcula o grau total de um vértice (indegree + outdegree)"""
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    return self.indegree(vertex) + self.outdegree(vertex)

  def degree_centrality(self, vertex):
    """Calcula a centralidade de grau para grafo direcionado"""
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    if self.order <= 1:
      return 0.0

    # Para grafo direcionado: grau máximo = 2*(n-1)
    max_possible_degree = 2 * (self.order - 1)
    return self.degree(vertex) / max_possible_degree



  def analyze_degree_centrality(self, vertex=None, show_details=True):
      print(f"\n=== ANÁLISE DE CENTRALIDADE DE GRAU - GRAFO DIRECIONADO ===")
      print(f"Total de vértices: {self.order}")
      print(f"Total de arestas: {self.size}")

      results = {}

      if vertex is not None:
          # analisa apenas um vértice
          if vertex not in self.vertices:
              print(f"Erro: Vértice '{vertex}' não existe!")
              return results

          centrality = self.degree_centrality(vertex)
          out_deg = self.outdegree(vertex)
          in_deg = self.indegree(vertex)
          total_deg = self.degree(vertex)

          results[vertex] = centrality

          print(f"\n Análise do vértice '{vertex}':")
          print(f"   Out-degree: {out_deg}")
          print(f"   In-degree: {in_deg}")
          print(f"   Grau total: {total_deg}")
          print(f"   Centralidade: {centrality:.4f}")

          if show_details:
              max_possible = 2 * (self.order - 1)
              print(f"   Cálculo: {total_deg} / {max_possible} = {centrality:.4f}")

      else:
          for v in sorted(self.vertices):
              centrality = self.degree_centrality(v)
              out_deg = self.outdegree(v)
              in_deg = self.indegree(v)
              total_deg = self.degree(v)

              results[v] = centrality
              #print(f"{v:<15} {out_deg:<8} {in_deg:<8} {total_deg:<8} {centrality:<12.4f}")

          # Estatísticas
          if results:
              max_vertex = max(results, key=results.get)
              min_vertex = min(results, key=results.get)
              avg_centrality = sum(results.values()) / len(results)

              print(f"\n Estatísticas:")
              print(f"   Maior centralidade: '{max_vertex}' ({results[max_vertex]:.4f})")
              print(f"   Menor centralidade: '{min_vertex}' ({results[min_vertex]:.4f})")
              print(f"   Centralidade média: {avg_centrality:.4f}")

      return results

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
    global_visited = []
    nodes = list(self.body)
    components = 0
    while len(global_visited) != len(self.body):
      node = random.choice(nodes)
      visited = self.dfs_iterative(node, global_visited)
      global_visited += visited
      nodes = list(set(nodes) - set(visited))
      if visited:
        components += 1

    print(f"O número de componentes é: {components}")

  def has_edge(self, vertex1, vertex2):
    """Verifica se existe uma aresta entre vertex1 e vertex2 (não-direcionado)"""
    if vertex1 not in self.vertices or vertex2 not in self.vertices:
      return False
    # Em grafo não-direcionado, verifica ambas as direções
    return vertex2 in self.body[vertex1] or vertex1 in self.body[vertex2]

  def indegree(self, vertex):
    """Para grafo não-direcionado, indegree = degree"""
    return self.degree(vertex)

  def outdegree(self, vertex):
    """Para grafo não-direcionado, outdegree = degree"""
    return self.degree(vertex)

  def degree(self, vertex):
    """Calcula o grau de um vértice em grafo não-direcionado"""
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    # Conta todas as conexões únicas
    connections = set()

    # Adiciona conexões onde vertex é origem
    for neighbor in self.body[vertex]:
      connections.add(neighbor)

    # Adiciona conexões onde vertex é destino
    for v in self.vertices:
      if v != vertex and vertex in self.body[v]:
        connections.add(v)

    return len(connections)
        
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
  def degree_centrality(self, vertex):
      """Calcula a centralidade de grau para grafo não-direcionado"""
      if vertex not in self.vertices:
        raise ValueError("Vértice não existe!")

      if self.order <= 1:
        return 0.0

      # Para grafo não-direcionado: grau máximo = (n-1)
      max_possible_degree = self.order - 1
      return self.degree(vertex) / max_possible_degree
  
  def analyze_degree_centrality(self, vertex=None, show_details=True):

    print(f"\n=== ANÁLISE DE CENTRALIDADE DE GRAU - GRAFO NÃO-DIRECIONADO ===")
    print(f"Total de vértices: {self.order}")
    print(f"Total de arestas: {self.size}")

    results = {}

    if vertex is not None:
        if vertex not in self.vertices:
            print(f"Erro: Vértice '{vertex}' não existe!")
            return results

        centrality = self.degree_centrality(vertex)
        deg = self.degree(vertex)

        results[vertex] = centrality

        print(f"\n Análise do vértice '{vertex}':")
        print(f"   Grau: {deg}")
        print(f"   Centralidade: {centrality:.4f}")

        if show_details:
            max_possible = self.order - 1
            print(f"   Cálculo: {deg} / {max_possible} = {centrality:.4f}")

    else:
        for v in sorted(self.vertices):
            centrality = self.degree_centrality(v)
            deg = self.degree(v)

            results[v] = centrality
            #print(f"{v:<15} {deg:<8} {centrality:<12.4f}")

        # Estatísticas
        if results:
            max_vertex = max(results, key=results.get)
            min_vertex = min(results, key=results.get)
            avg_centrality = sum(results.values()) / len(results)

            print(f"\n Estatísticas:")
            print(f"   Maior centralidade: '{max_vertex}' ({results[max_vertex]:.4f})")
            print(f"   Menor centralidade: '{min_vertex}' ({results[min_vertex]:.4f})")
            print(f"   Centralidade média: {avg_centrality:.4f}")

    return results


  def minimum_spannig_tree(self, vertex):
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")
    visited = []
    stack = []

    stack.append(vertex)

    while len(stack) > 0:
      element = stack.pop()

      if element not in visited:
        visited.append(element)

        for adj in self.body[element]:
          if adj not in visited:
            stack.append(adj)
    sub_graph ={}
    for i in visited:
      sub_graph[i] = {}
      for adj, weight in self.body[i].items():
          if adj in visited:
              sub_graph[i][adj] = weight

    #ALGORITMO DE PRIM
    total_cost = 0
    MST = {vertex: {}}

    while len(MST) < len(sub_graph):
        lower_weight = np.inf
        for source_node in MST.keys():
            for destination_node, weight in sub_graph[source_node].items():
                if destination_node not in MST and weight < lower_weight:
                    lower_node = destination_node
                    lower_weight = weight
                    source = source_node
            if lower_node is None:
              break;

        MST[lower_node] = {source: lower_weight}
        MST[source][lower_node] = lower_weight
        total_cost += lower_weight

    return  MST, total_cost

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
      director = format_name(director)
      if director not in graph_d.vertices:
        graph_d.add_vertex(director)
    
      #add cada um dos vértices de atores
    cast = return_values(cast)
    for actor in cast:
      actor = format_name(actor)
      if actor not in graph_d.vertices:
        graph_d.add_vertex(actor)
        graph_u.add_vertex(actor)

    for actor in cast:
      actor = format_name(actor)
      work_together_actor = work_together(actor, cast)
      for a in work_together_actor:
        a = format_name(a)
        weight = graph_u.get_weight(actor, a)
        if weight:
          graph_u.add_edge(actor, a, weight + 1)  # Adiciona com peso incrementado
        else:
          graph_u.add_edge(actor, a, 1)  # Primeira colaboração

    #add as arestas ponderadas
    for director in directors:
      for actor in cast:
        actor = format_name(actor)
        director = format_name(director)
        weight = graph_u.get_weight(actor, a)
        if weight:
          graph_d.add_edge(actor, director, weight + 1)  # Adiciona com peso incrementado
        else:
          graph_d.add_edge(actor, director, 1)  # Primeira colaboração

  return graph_d, graph_u

def read_graph_csv(csv, graph):
  df = pd.read_csv(csv)

  for _, row in df.iterrows():
    try:
      graph.add_edge(row['Origem'], row['Destino'], row['Peso'])
    except:
       pass

  return graph

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

def format_name(name):
  name = name.split(" ")
  final_name = ''
  for n in name:
    final_name += n.capitalize()
  return final_name.replace(' ', "")#garante que tira os espaços