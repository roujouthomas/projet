#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:04:37 2024

@author: thomasroujou
"""

import numpy as np

def marchealéatoire(p, N):
    pas = np.random.choice([-1, 1], size=N, p=[1-p, p])
    deplacement = np.sum(pas)
    return deplacement

# Exemple 
p = 0.6
N = 100
deplacement = marchealéatoire(p, N)
print(f"Déplacement après {N} pas avec p = {p} : {deplacement}")

def monte_carlo(p, N, M):
    deplacement = []
    for i in range(M):
        deplacement.append(marchealéatoire(p, N))
    deplacement = np.array(deplacement)
    moyenne = np.mean(deplacement)
    écart_type = np.std(deplacement)
    return deplacement, moyenne, écart_type


M = 1000
deplacement, moyenne, écart_type = monte_carlo(p, N, M)
print(f"Moyenne des déplacements pour M = {M} : {moyenne}")
print(f"Écart-type des déplacements pour M = {M} : {écart_type}")

def montecarloanalyse(p, N, Mlist):
    resultas = np.zeros((2, len(Mlist) + 1))
    moyenne_theorique = N * (2 * p - 1)
    écart_type_theorique = np.sqrt(N * 4 * p * (1 - p))
    
    for i, M in enumerate(Mlist):
        deplacement, moyenne, écart_type = monte_carlo(p, N, M)
        resultas[0, i] = moyenne
        resultas[1, i] = écart_type
    
    resultas[0, -1] = moyenne_theorique
    resultas[1, -1] = écart_type_theorique
    
    return resultas

# Exemple : p = 0.6, N = 100, M_values = [10, 100, 1000, 10000]
M_values = [10, 100, 1000, 10000]
resultas = montecarloanalyse(p, N, M_values)

print("Tableau des résultats (lignes : moyennes, écart-types, colonne finale : valeurs théoriques) :")
print(resultas)


np.save("montecarloanalyse.npy", resultas)