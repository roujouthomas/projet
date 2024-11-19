#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:01:40 2024

@author: thomasroujou
"""

import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'algorithme
P = 30        # Nombre de conteneurs
N = 3         # Masse, Volume, Prix
K = 10000     # Nombre de générations
M = 5        # Taille des tournois
Pmut = 0.5    # Probabilité de mutation
Dmut = 0.01   # Taille du pas de mutation

v = np.array([
    [53, 44, 5],
    [66, 32, 5],
    [17, 32, 3],
    [21, 40, 6],
    [15, 10, 3],
    [47, 37, 1],
    [6, 38, 6],
    [81, 10, 7],
    [78, 22, 5],
    [76, 27, 5],
    [85, 37, 1],
    [99, 20, 2],
    [18, 24, 9],
    [8, 33, 9],
    [72, 37, 9],
    [76, 8, 8],
    [98, 12, 5],
    [85, 14, 6],
    [86, 9, 5],
    [71, 11, 8],
    [50, 47, 10],
    [18, 23, 6],
    [46, 40, 4],
    [54, 33, 8],
    [50, 11, 9],
    [1, 39, 4],
    [26, 19, 1],
    [22, 28, 3],
    [25, 9, 1],
    [8, 2, 1]
])  # Matrice des valeurs : Masse, Volume, Prix

# Fonction objectif à maximiser
def f(x):
    calc = v * x[:, np.newaxis]  # Appliquer x à chaque ligne de v
    sommeP = np.sum(calc[:, 0])  # Somme des masses
    sommeV = np.sum(calc[:, 1])  # Somme des volumes
    sommePrix = np.sum(calc[:, 2])  # Somme des prix

    # Vérification stricte des contraintes de masse et volume
    if sommeP <= 800 and sommeV <= 600:
        return sommeP  # Maximiser le prix total
    else:
        return 0  # Solution non viable si les contraintes ne sont pas respectées

# Sélection par tournoi
def tournament(A, m):
    p, n = A.shape  # Taille de la matrice A
    B = np.zeros((p, m))  # Matrice pour stocker les individus sélectionnés

    # Sélectionner m individus aléatoirement
    for i in range(m):
        B[:, i] = A[:, np.random.randint(n)]  # Sélection d'un individu aléatoire

    # Trier les individus sélectionnés par leur fitness (dernière ligne)
    A1t = B[:, B[-1, :].argsort()[::-1]]  # Tri par ordre décroissant de la dernière ligne (fitness)
    A1 = A1t[:, 0]  # Garder le premier, c'est-à-dire le meilleur
    return A1

# Fonction de croisement entre deux individus
def crossover(A1, A2, p):
    A = A2[:p].copy()  # Copier uniquement les gènes de A2 (pas la fitness)
    q = np.random.randint(1, p+1)  # Générer un entier aléatoire entre 1 et p
    A[:q] = A1[:q]  # Remplacer les q premiers éléments de A2 par ceux de A1
    return A

# Fonction de mutation
def mutation(X, p, Pmut, dmut):
    mut = np.random.rand(p) < Pmut  # vecteur booléen : True si mutation, False sinon
    A = X + dmut * mut * np.random.randn(p)  # modification avec une loi normale
    A = np.clip(A, 0.0, 1.0)  # remettre les valeurs dans l'intervalle [0,1]
    return A

# Initialisation du génome aléatoire dans [0,1]
A = np.random.rand(P+1, N)  # N colonnes : génome + fitness
fbest = -1E90
fplot = np.zeros(K)

# Boucle sur les générations
for i in range(K):
    # Évaluation de la fonction pour tous les individus
    for j in range(N):
        A[P, j] = f(A[:P, j])  # Fitness de chaque individu

    # Seuil binaire des valeurs
    A[:P, :] = np.where(A[:P, :] < 0.5, 0, 1)
    
    # Tri par ordre décroissant de la fitness
    A = A[:, A[P, :].argsort()[::-1]]
    
    # Mise à jour du meilleur individu
    if A[P, 0] > fbest:
        fbest = A[P, 0]
        Abest = A[:P, 0]

        print(f"Generation: {i}    f_max: {fbest:.6f}    Genome: {Abest}")

        # Plot du meilleur individu
        fplot[i] = fbest

    fplot[i] = fbest
    
    # Sélection et croisement pour remplacer les N/2 moins bons
    Aold = A.copy()
    for j in range(N//2, N):
        A1 = tournament(Aold, M)
        A2 = tournament(Aold, M)
        A[:P, j] = crossover(A1, A2, P)  # Seules les P premières lignes (gènes)

    # Mutation sauf pour le meilleur
    for j in range(1, N):
        A[:P, j] = mutation(A[:P, j], P, Pmut, Dmut)

# Tracé final
plt.figure(2)
plt.loglog(np.arange(1, K+1)*N, fplot, 'k-')
plt.xlim([1, K*N])
plt.xlabel('Number of evaluations')
plt.ylabel('Fitness')
plt.show()

