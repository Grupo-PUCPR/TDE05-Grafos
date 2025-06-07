import pandas as pd
import numpy as np
from graph import *

# Lê o arquivo CSV
df = pd.read_csv('netflix_amazon_disney_titles.csv')

# Cria uma instância do grafo direcionado
g = Graph_directed()

# Constrói o grafo com os dados
g.construct_graph(df)
