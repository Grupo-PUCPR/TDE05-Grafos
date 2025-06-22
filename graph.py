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
        # "Limpa" o nome do vértice de origem para ser compatível com a saída
        safe_origem = origem.encode('cp1252', 'replace').decode('cp1252')
        
        # "Limpa" o nome de cada vértice de destino antes de juntá-los na string
        safe_adjacentes_list = []
        for destino, peso in destinos.items():
            safe_destino = destino.encode('cp1252', 'replace').decode('cp1252')
            safe_adjacentes_list.append(f'{safe_destino}({peso})')
            
        adjacentes_str = ', '.join(safe_adjacentes_list)
        
        result += f'{safe_origem} -> {adjacentes_str}\n'
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
    if node not in self.vertices:
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
  
  def bfs_shortest_path(self, start_node):
    if start_node not in self.vertices:
      return {}

    distances = {node: float('inf') for node in self.vertices}
    distances[start_node] = 0

    queue = [start_node]

    while queue:
        current_node = queue.pop(0)

        for neighbor in self.get_adjacent(current_node):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)

    return distances

  def dijkstra(self, source_vertex):
    """
    Implementa o algoritmo de Dijkstra para encontrar caminhos mais curtos

    Args:
        source_vertex: Vértice de origem

    Returns:
        dict: {vertex: (distance, predecessor)} para cada vértice
    """
    if source_vertex not in self.vertices:
      raise ValueError("Vértice de origem não existe!")

    # Inicialização
    distances = {vertex: float('inf') for vertex in self.vertices}
    predecessors = {vertex: None for vertex in self.vertices}
    distances[source_vertex] = 0

    # Conjunto de vértices não visitados
    unvisited_vertices = set(self.vertices)

    while unvisited_vertices:
      # Encontra o vértice não visitado com menor distância
      current_vertex = min(unvisited_vertices, key=lambda vertex: distances[vertex])

      # Se a distância é infinita, não há mais vértices alcançáveis
      if distances[current_vertex] == float('inf'):
        break

      # Remove o vértice atual dos não visitados
      unvisited_vertices.remove(current_vertex)

      # Atualiza distâncias dos vizinhos
      for neighbor_vertex in self.get_adjacent(current_vertex):
        if neighbor_vertex in unvisited_vertices:
          edge_weight = self.get_weight(current_vertex, neighbor_vertex)
          if edge_weight is not False:
            alternative_distance = distances[current_vertex] + edge_weight

            if alternative_distance < distances[neighbor_vertex]:
              distances[neighbor_vertex] = alternative_distance
              predecessors[neighbor_vertex] = current_vertex

    # Retorna no formato esperado: {vertex: (distance, predecessor)}
    return {vertex: (distances[vertex], predecessors[vertex]) for vertex in self.vertices}

  def find_all_shortest_paths(self, source_vertex, destination_vertex):
    """
    Encontra todos os caminhos mais curtos entre dois vértices

    Args:
        source_vertex: Vértice de origem
        destination_vertex: Vértice de destino

    Returns:
        list: Lista de todos os caminhos mais curtos (cada caminho é uma lista de vértices)
    """
    if source_vertex not in self.vertices or destination_vertex not in self.vertices:
      return []

    if source_vertex == destination_vertex:
      return [[source_vertex]]

    # Executa Dijkstra a partir da origem
    shortest_paths_from_source = self.dijkstra(source_vertex)

    # Se o destino não é alcançável, retorna lista vazia
    if shortest_paths_from_source[destination_vertex][0] == float('inf'):
      return []

    # Distância mínima até o destino
    min_distance = shortest_paths_from_source[destination_vertex][0]

    # Função recursiva para encontrar todos os caminhos
    def backtrack_all_paths(current_vertex, current_path, target_distance):
      """
      Usa backtracking para encontrar todos os caminhos ótimos

      Args:
          current_vertex: Vértice atual no backtracking
          current_path: Caminho atual sendo construído
          target_distance: Distância que deve ser alcançada

      Returns:
          list: Lista de caminhos válidos
      """
      if current_vertex == source_vertex and target_distance == 0:
        return [current_path[::-1]]  # Inverte o caminho

      if target_distance <= 0:
        return []

      all_paths = []

      # Verifica todos os possíveis predecessores
      for predecessor_vertex in self.vertices:
        if predecessor_vertex != current_vertex:
          edge_weight = self.get_weight(predecessor_vertex, current_vertex)
          if edge_weight is not False:
            # Verifica se este predecessor leva a um caminho ótimo
            predecessor_distance = shortest_paths_from_source[predecessor_vertex][0]
            if predecessor_distance + edge_weight == shortest_paths_from_source[current_vertex][0]:
              new_path = current_path + [predecessor_vertex]
              paths_from_predecessor = backtrack_all_paths(
                predecessor_vertex,
                new_path,
                target_distance - edge_weight
              )
              all_paths.extend(paths_from_predecessor)

      return all_paths

    # Inicia o backtracking a partir do destino
    all_shortest_paths = backtrack_all_paths(destination_vertex, [destination_vertex], min_distance)

    return all_shortest_paths

  def betweenness_centrality(self, target_vertex):
    """
    Calcula a centralidade de intermediação para um vértice específico

    Args:
        target_vertex: Vértice para o qual calcular a centralidade

    Returns:
        float: Valor de centralidade de intermediação bruta (sem normalização)
    """
    if target_vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    # Casos especiais
    if self.order <= 2:
      return 0.0

    betweenness_score = 0.0

    # Para cada par de vértices (source, destination)
    for source_vertex in self.vertices:
      for destination_vertex in self.vertices:
        # Ignora casos onde source = target ou target = destination
        if source_vertex != target_vertex and destination_vertex != target_vertex and source_vertex != destination_vertex:

          # Encontra todos os caminhos mais curtos entre source e destination
          all_shortest_paths = self.find_all_shortest_paths(source_vertex, destination_vertex)

          if all_shortest_paths:
            total_shortest_paths = len(all_shortest_paths)
            paths_through_target = 0

            # Conta quantos caminhos passam pelo target_vertex
            for path in all_shortest_paths:
              if target_vertex in path[1:-1]:  # Exclui extremos
                paths_through_target += 1

            # Adiciona a contribuição deste par à centralidade
            if total_shortest_paths > 0:
              betweenness_score += paths_through_target / total_shortest_paths

    # Retorna valor bruto - normalização será feita nas subclasses
    return betweenness_score

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

  def closeness_centrality(self, vertex = None):
    raise NotImplementedError("Tem que ser implementado na subclasse!")

  def analyze_closeness_centrality(self):
    print(f"\n=== ANÁLISE DE CENTRALIDADE DE PROXIMIDADE ===")
    print(f"Tipo de Grafo: {self.__class__.__name__}")
    
    results = {}
    
    for v in self.vertices:
      results[v] = self.closeness_centrality(v)


    print("\nCentralidade de cada vértice:")
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    for vertex, centrality in sorted_results[:10]:
        safe_vertex_name = vertex.encode('cp1252', 'replace').decode('cp1252')
        print(f"  - {safe_vertex_name:<20}: {centrality:.8f}")
        # < 20 formatar com mesmo numero char todos os nomes

    if results:
        max_vertex = max(results, key=results.get)
        min_vertex = min(results, key=results.get)
        avg_centrality = sum(results.values()) / len(results)

        print(f"\nEstatísticas:")
        print(f"  - Maior centralidade: '{max_vertex}' ({results[max_vertex]:.8f})")
        print(f"  - Menor centralidade: '{min_vertex}' ({results[min_vertex]:.8f})")
        print(f"  - Centralidade média: {avg_centrality:.8f}")

    return results

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

    # Verifica se existe aresta direcionada de vertex1 para vertex2
    if vertex1 in self.body and vertex2 in self.body[vertex1]:
      return self.body[vertex1][vertex2]

    return False
  
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

  def closeness_centrality(self, vertex = None):
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    distances = self.bfs_shortest_path(vertex)
    
    sum_of_inverse_distances = 0
    n_reachable_nodes = 0

    for node, dist in distances.items():
        if dist > 0 and dist != float('inf'):
            sum_of_inverse_distances += 1 / dist
            n_reachable_nodes += 1

    if n_reachable_nodes == 0:
        return 0
    
    return sum_of_inverse_distances / n_reachable_nodes
    
  def analyze_degree_centrality(self, vertex=None, show_details=False):
    """
    Analisa a centralidade de grau de forma otimizada, evitando recálculos.
    """
    print(f"\n=== ANÁLISE DE CENTRALIDADE DE GRAU - GRAFO DIRECIONADO ===")
    print(f"Total de vértices: {self.order}")
    print(f"Total de arestas: {self.size}")

    results = {}

    if self.order <= 1:
        max_possible_degree = 1
    else:
        max_possible_degree = 2 * (self.order - 1)


    if vertex is not None:
        # --- Análise de um único vértice ---
        if vertex not in self.vertices:
            print(f"Erro: Vértice '{vertex}' não existe!")
            return results

        out_deg = self.outdegree(vertex)
        in_deg = self.indegree(vertex)
        
        total_deg = out_deg + in_deg
        
        centrality = total_deg / max_possible_degree
        
        results[vertex] = centrality

        print(f"\n Análise do vértice '{vertex}':")
        print(f"   Out-degree: {out_deg}")
        print(f"   In-degree: {in_deg}")
        print(f"   Grau total: {total_deg}")
        print(f"   Centralidade: {centrality:.4f}")

        if show_details:
            print(f"   Cálculo: {total_deg} / {max_possible_degree} = {centrality:.4f}")

    else:
        # --- Análise de todos os vértices ---
        for v in sorted(self.vertices):
            out_deg = self.outdegree(v)
            in_deg = self.indegree(v)
            total_deg = out_deg + in_deg
            centrality = total_deg / max_possible_degree
            
            results[v] = centrality
        
        if results:
            max_vertex = max(results, key=results.get)
            min_vertex = min(results, key=results.get)
            avg_centrality = sum(results.values()) / len(results)

            print(f"\n Estatísticas:")
            print(f"   Maior centralidade: '{max_vertex}' ({results[max_vertex]:.4f})")
            print(f"   Menor centralidade: '{min_vertex}' ({results[min_vertex]:.4f})")
            print(f"   Centralidade média: {avg_centrality:.4f}")

    return results

  def betweenness_centrality(self, target_vertex):
    """
    Calcula a centralidade de intermediação para grafo direcionado
    """
    # Chama o método da classe base para calcular a centralidade bruta
    raw_betweenness_score = super().betweenness_centrality(target_vertex)

    # Normalização para grafo direcionado: (n-1) × (n-2)
    max_possible_betweenness = (self.order - 1) * (self.order - 2)

    if max_possible_betweenness > 0:
      return raw_betweenness_score / max_possible_betweenness
    else:
      return 0.0

  def analyze_betweenness_centrality(self):
    """
    Analisa a centralidade de intermediação para todos os vértices do grafo direcionado

    Returns:
        dict: {vertex: centrality_value} para todos os vértices
    """
    print(f"\n=== ANÁLISE DE CENTRALIDADE DE INTERMEDIAÇÃO - GRAFO DIRECIONADO ===")
    print(f"Total de vértices: {self.order}")
    print(f"Total de arestas: {self.size}")

    centrality_results = {}

    # Calcula centralidade para cada vértice
    for vertex in self.vertices:
      vertex_centrality = self.betweenness_centrality(vertex)
      centrality_results[vertex] = vertex_centrality

    # Exibe resultados ordenados
    print("\nCentralidade de Intermediação de cada vértice:")
    sorted_results = sorted(centrality_results.items(), key=lambda item: item[1], reverse=True)
    for vertex, centrality in sorted_results:
      print(f"  - {vertex:<20}: {centrality:.4f}")

    # Estatísticas
    if centrality_results:
      highest_centrality_vertex = max(centrality_results, key=centrality_results.get)
      lowest_centrality_vertex = min(centrality_results, key=centrality_results.get)
      average_centrality = sum(centrality_results.values()) / len(centrality_results)

      print(f"\nEstatísticas:")
      print(f"  - Maior centralidade: '{highest_centrality_vertex}' ({centrality_results[highest_centrality_vertex]:.4f})")
      print(f"  - Menor centralidade: '{lowest_centrality_vertex}' ({centrality_results[lowest_centrality_vertex]:.4f})")
      print(f"  - Centralidade média: {average_centrality:.4f}")

    return centrality_results

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

    # Para grafo não-direcionado, adiciona aresta em ambas as direções
    self.body[vertex1][vertex2] = weight
    self.body[vertex2][vertex1] = weight
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

    # Verifica se existe aresta direta
    if vertex1 in self.body and vertex2 in self.body[vertex1]:
      return self.body[vertex1][vertex2]

    # Para grafo não-direcionado, verifica também a direção inversa
    if vertex2 in self.body and vertex1 in self.body[vertex2]:
      return self.body[vertex2][vertex1]

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
    """
    Analisa a centralidade de grau para grafo não-direcionado.
    """
    print(f"\n=== ANÁLISE DE CENTRALIDADE DE GRAU - GRAFO NÃO-DIRECIONADO ===")
    print(f"Total de vértices: {self.order}")
    print(f"Total de arestas: {self.size}")

    results = {}

    if self.order <= 1:
        max_possible_degree = 1 
    else:
        max_possible_degree = self.order - 1

    if vertex is not None: #apenas um vertice
        if vertex not in self.vertices:
            print(f"Erro: Vértice '{vertex}' não existe!")
            return results

        deg = self.degree(vertex)
        centrality = deg / max_possible_degree

        results[vertex] = centrality

        print(f"\n Análise do vértice '{vertex}':")
        print(f"   Grau: {deg}")
        print(f"   Centralidade: {centrality:.4f}")

        if show_details:
            # Reutiliza a variável 'deg'
            print(f"   Cálculo: {deg} / {max_possible_degree} = {centrality:.4f}")

    else: #todos os vertices
        for v in sorted(self.vertices):
            deg = self.degree(v)
            centrality = deg / max_possible_degree
            results[v] = centrality

        print("\nCentralidade de cada vértice (Top 10):")
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        for v_name, cent in sorted_results[:10]:
            safe_vertex_name = v_name.encode('cp1252', 'replace').decode('cp1252')
            print(f"  - {safe_vertex_name:<20}: {cent:.4f}")

        if results:
            max_vertex = max(results, key=results.get)
            min_vertex = min(results, key=results.get)
            avg_centrality = sum(results.values()) / len(results)

            print(f"\n Estatísticas:")
            print(f"   Maior centralidade: '{max_vertex}' ({results[max_vertex]:.4f})")
            print(f"   Menor centralidade: '{min_vertex}' ({results[min_vertex]:.4f})")
            print(f"   Centralidade média: {avg_centrality:.4f}")

    return results
  
  # Função que calcula a Centralidade de Proximidade
  def closeness_centrality(self, vertex):
    if vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    distances = self.bfs_shortest_path(vertex)

    sum_of_distances = 0
    n_reachable_nodes = 0

    for node, dist in distances.items():
        if dist > 0 and dist != float('inf'):
            sum_of_distances += dist
            n_reachable_nodes += 1

    if n_reachable_nodes == 0:
      return 0

    centrality = n_reachable_nodes / sum_of_distances
    
    return centrality

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

  def betweenness_centrality(self, target_vertex):
    """
    Calcula a centralidade de intermediação para grafo não-direcionado
    """
    if target_vertex not in self.vertices:
      raise ValueError("Vértice não existe!")

    # Casos especiais
    if self.order <= 2:
      return 0.0

    betweenness_score = 0.0

    # Para grafo não-direcionado, evita contar pares duplicados
    # Usa apenas pares onde source_vertex < destination_vertex lexicograficamente
    vertices_sorted = sorted(self.vertices)

    for i, source_vertex in enumerate(vertices_sorted):
      for j, destination_vertex in enumerate(vertices_sorted):
        if i < j:  # Evita pares duplicados e self-loops
          # Ignora casos onde source = target ou target = destination
          if source_vertex != target_vertex and destination_vertex != target_vertex:

            # Encontra todos os caminhos mais curtos entre source e destination
            all_shortest_paths = self.find_all_shortest_paths(source_vertex, destination_vertex)

            if all_shortest_paths:
              total_shortest_paths = len(all_shortest_paths)
              paths_through_target = 0

              # Conta quantos caminhos passam pelo target_vertex
              for path in all_shortest_paths:
                if target_vertex in path[1:-1]:  # Exclui extremos
                  paths_through_target += 1

              # Adiciona a contribuição deste par à centralidade
              if total_shortest_paths > 0:
                betweenness_score += paths_through_target / total_shortest_paths

    # Normalização para grafo não-direcionado: (n-1) × (n-2) / 2
    max_possible_betweenness = (self.order - 1) * (self.order - 2) / 2

    if max_possible_betweenness > 0:
      return betweenness_score / max_possible_betweenness
    else:
      return 0.0

  def analyze_betweenness_centrality(self):
    """
    Analisa a centralidade de intermediação para todos os vértices do grafo não-direcionado

    Returns:
        dict: {vertex: centrality_value} para todos os vértices
    """
    print(f"\n=== ANÁLISE DE CENTRALIDADE DE INTERMEDIAÇÃO - GRAFO NÃO-DIRECIONADO ===")
    print(f"Total de vértices: {self.order}")
    print(f"Total de arestas: {self.size}")

    centrality_results = {}

    # Calcula centralidade para cada vértice
    for vertex in self.vertices:
      vertex_centrality = self.betweenness_centrality(vertex)
      centrality_results[vertex] = vertex_centrality

    # Exibe resultados ordenados
    print("\nCentralidade de Intermediação de cada vértice:")
    sorted_results = sorted(centrality_results.items(), key=lambda item: item[1], reverse=True)
    for vertex, centrality in sorted_results:
      print(f"  - {vertex:<20}: {centrality:.4f}")

    # Estatísticas
    if centrality_results:
      highest_centrality_vertex = max(centrality_results, key=centrality_results.get)
      lowest_centrality_vertex = min(centrality_results, key=centrality_results.get)
      average_centrality = sum(centrality_results.values()) / len(centrality_results)

      print(f"\nEstatísticas:")
      print(f"  - Maior centralidade: '{highest_centrality_vertex}' ({centrality_results[highest_centrality_vertex]:.4f})")
      print(f"  - Menor centralidade: '{lowest_centrality_vertex}' ({centrality_results[lowest_centrality_vertex]:.4f})")
      print(f"  - Centralidade média: {average_centrality:.4f}")

    return centrality_results

def work_together(actor, cast):
    actor = format_name(actor)
    cast = [format_name(a) for a in cast]
    return set(cast) - {actor}


#demora pq ele itera por tudo
def construct_graph(graph_d, graph_u, df):
  df = df.dropna().copy() #já tiro todas as linhas que não tiverem um valor (NaN)
  df['director'] = df['director'].apply(lambda x: [format_name(d) for d in return_values(str(x))])
  df['cast'] = df['cast'].apply(lambda x: [format_name(c) for c in return_values(str(x))])
  for title, directors, cast in df.values:
    # As 4 linhas abaixo foram REMOVIDAS pois eram redundantes e causavam o erro.
    # directors = str(directors)
    # cast = str(cast)
    # directors = return_values(directors)
    # cast = return_values(cast)
  
    # O resto do código funciona perfeitamente com as listas 'directors' e 'cast'
    for director in directors:
      if director and director not in graph_d.vertices: # Adicionado 'if director' para segurança
        graph_d.add_vertex(director)
    
    for actor in cast:
      if actor and actor not in graph_d.vertices: # Adicionado 'if actor' para segurança
        graph_d.add_vertex(actor)
        graph_u.add_vertex(actor)

    for actor in cast:
      if not actor: continue # Pula se o ator for uma string vazia
      work_together_actor = work_together(actor, cast)
      for a in work_together_actor:
        if not a: continue # Pula se o outro ator for uma string vazia
        weight = graph_u.get_weight(actor, a)
        graph_u.add_edge(actor, a, (weight or 0) + 1)

    for director in directors:
      if not director: continue
      for actor in cast:
        if not actor: continue # Pula se o ator for uma string vazia
        # A lógica para arestas direcionadas parece um pouco estranha (ator -> diretor com peso de colaboração)
        # Mantendo como está, mas pode ser um ponto de atenção.
        # Assumindo que o peso aqui deve ser baseado em quantas vezes trabalharam juntos.
        weight = graph_d.get_weight(actor, director)
        graph_d.add_edge(actor, director, (weight or 0) + 1)
        
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
      origem=format_name(origem)
      for destino, peso in destinos.items():
          destino = format_name(destino)
          data.append((origem, destino, peso))

  df = pd.DataFrame(data, columns=['Origem', 'Destino', 'Peso'])
  if transpose:
      df.to_csv(f'graph_{graph.__class__.__name__}_transpose.csv', index=False)
  else:
    df.to_csv(f'graph_{graph.__class__.__name__}.csv', index=False)

def format_name(name):
    if pd.isnull(name):
        return ''
    name = str(name)  # Garante que é string
    name = name.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
    name = name.replace(' ', '').upper()
    return name

def return_values(value):
    if pd.isnull(value) or value == '':
        return []
    return [v.strip() for v in value.split(',')]
