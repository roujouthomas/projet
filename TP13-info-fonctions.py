## TP 13

# Modules
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Fonctions affichages
def affichage_graphe(G):
    '''fonction affichant le graphe représenté par un dictionnaire ou un tableau numpy G'''
    plt.clf()
    Gx = nx.DiGraph(G)
    nx.draw_networkx(Gx)
    plt.show()

def draw_graph(G):
    '''fonction affichant le graphe représenté par une liste de listes G'''
    plt.clf()
    G_ = nx.DiGraph()
    for i in range(len(G)):
        for j in G[i]:
            G_.add_edge(i, j)
    nx.draw_networkx(G_, font_color ="w", node_color="black", node_size=600, arrowsize=35, font_size=16)
    plt.show()

def draw_count(G, rank):
    '''fonction affichant le graphe représenté par une liste de listes G, avec pour chaque sommet, un disque de rayon proportionnel à son rang donné par la liste rank'''
    plt.clf()
    G_ = nx.DiGraph()
    G_.add_nodes_from(range(len(G)))
    for i in range(len(G)):
        for j in G[i]:
            G_.add_edge(i, j)
    r = [0]*len(rank)
    for i, e in enumerate(rank):
        r[e] = i + 1
    max_size = 5000
    a = (max_size - 500)/(1 - len(rank))
    node_sizes= [int(r[i]*a + max_size - a) for i in range(len(r))]
    nx.draw_networkx(G_, font_color ="w", node_color="black", node_size=node_sizes, arrowsize=35, font_size=16)
    plt.show()


# Représentations du graphe orienté donné en exemple
D = {
    0: [0, 1, 3], 
    1: [2], 
    2: [3], 
    3: [4], 
    4: [5], 
    5: [1]
    }

M = np.array([ 
    [1, 1, 0, 1, 0, 0], 
    [0, 0, 1, 0, 0, 0], 
    [0, 0, 0, 1, 0, 0], 
    [0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 1], 
    [0, 1, 0, 0, 0, 0] 
    ])

