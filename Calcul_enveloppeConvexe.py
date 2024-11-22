#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:04:10 2024

@author: thomasroujou
"""

import random
import random
import math
from math import *
import matplotlib.pyplot as plt
from typing import List, Tuple


def occ(x,S):
    nb=0
    for i in range(len(S)):
        if S[i]==x:
            nb+=1
    return nb
#O(n^2)
def bourrin(S):
    for x in S:
        if occ(x,S)>(len(S)/2):
             return x
    return None

#O(nlog(n))
def separation(liste):
    mid= len(liste) // 2
    partie1 = liste[:mid]
    partie2 = liste[mid:]  
    
    return partie1, partie2


def Maj(S):
    if S is None:
        return None
    if len(S)==1:
        return S[0]
    else:
        S1,S2=separation(S)
        x1=Maj(S1)
        x2=Maj(S2)
        if x1 != None and occ(x1,S) >len(S)//2:
            return x1
        if x1 != None and occ(x2,S) >len(S)//2:
            return x2
        return None


#O(n) en moyenne O(n^2) dans le pire des cas
def methodedico(S):
    dico={}
    print(len(S))
    for i in range(len(S)):
       if S[i] in dico:
           dico[S[i]]+=1
       else: 
           dico[S[i]]=1
    for x in dico:
        if dico[S[i]]>=len(S)//2: 
            return S[i]
    return dico

class Pile:
    def __init__(self):
        self.elements = []

    def empiler(self, x):
        self.elements.append(x)

    def depiler(self):
        if not self.est_vide():
            return self.elements.pop()
        return None

    def est_vide(self):
        return len(self.elements) == 0
#O(n)
def majoPile(S):
    P1 = Pile()
    P2 = Pile()
    for x in S:
        y1 = P1.depiler()
        P1.empiler(y1)
        if y1 is None or y1 != x: 
            P1.empiler(x)  
        else:
            y2 = P2.depiler()  
            if y2 == x:  
                P2.empiler(y2)  
                P2.empiler(x)   
            else:
                P1.empiler(y2)
                P1.empiler(x)  
    return(P1.elements,P2.elements)


L=[296, 173, 1, 739, 614, 454, 359, 255, 706, 816, 983, 586, 851, 570, 992, 506, 
812, 595, 905, 109, 303, 338, 359, 731, 77, 343, 307, 71, 795, 955, 91, 272, 960, 
324, 154, 306, 831, 354, 499, 741, 141,173, 185, 573, 3, 433, 778, 173, 470, 975, 
275, 912, 733, 735, 580, 719, 526, 187, 68, 323, 17, 263, 790, 777, 173, 163, 756, 
966, 129, 359, 168, 816, 939, 340, 852, 710, 84, 699, 173, 244, 579, 781, 35, 649, 
975, 209, 651, 718, 797, 359, 356, 182, 646, 682, 49, 173, 916, 432, 173, 707,3,1,3]
L2=[3,3,3,3,3,3,2,2,2,2]
def liste(n):
    L=[3,4,2,3,3,3,3,3,3,3]
    for i in range(n):
        L.append(3)
        L.append(random.randint(1,100))
    return L

############################################################################################################

#élément médian tableau
def tri_fusion(liste):
   
    if len(liste) <= 1:
        return liste
    milieu = len(liste) // 2
    gauche = liste[:milieu]
    droite = liste[milieu:]
    gauche = tri_fusion(gauche)
    droite = tri_fusion(droite)
    return fusion(gauche, droite)

def fusion(gauche, droite):
    resultat = []
    i = j = 0
    while i < len(gauche) and j < len(droite):
        if gauche[i] < droite[j]:
            resultat.append(gauche[i])
            i += 1
        else:
            resultat.append(droite[j])
            j += 1
    resultat.extend(gauche[i:])
    resultat.extend(droite[j:])
    
    return resultat
#O(nlog(n))
def median(S):
    a=len(S)//2
    L=tri_fusion(S)
    return L[a]

def separer5(L):
    n=len(L)
    M=[[0,0,0,0,0]for i in range ((n+4)//5 +1)]
    for i in range(n):
        M[i//5][i%5]=L[i]
    M[-1] = [x for x in M[-1] if x != 0]
    return M
#O(n)
def rang(S,k):
    if len(S) == 1:
        return 0
    A=separer5(S)
    for i in range(len(A)):
         A[i]=tri_fusion(A[i])
    L=[]
    for i in range(len(A)):
         if len(A[i]) >= 3:
             L.append(A[i][2])
    alpha= median(L)
    P=[S[i]<alpha for i in range(len(S))]
    G=[S[i]>alpha for i in range(len(S))]
    if k<len(P):
        return rang(P,k)
    if k==len(P)+1:
         return(alpha)
    return(rang(G,k-len(P)+1))


############################################################################################################
        
def calcul_angle_v2(pointref: Tuple[int, int], point2: Tuple[int, int]):
    dx = sqrt((pointref[0]-point2[0])**2)
    dy = sqrt((pointref[1]-point2[1])**2)
    distance = sqrt(dx**2+dy**2)

    if distance == 0:
        return float('inf')  #  éviter la division par zéro
    
    if point2[0] >= pointref[0] and point2[1] <= pointref[1] : # on est dans le quart bas a droite
        return acos(dy/distance)
    
    if point2[0]>= pointref[0] and point2[1] >= pointref[1] : # quart haut a droite
        return  acos(dx/distance) +pi/2

    if point2[0] <= pointref[0] and point2[1] >= pointref[1] : #quart haut gauche
        return  acos(dy/distance) + pi

    if point2[0] <= pointref[0] and point2[1] <= pointref[1] :
        return acos(dx/distance) + 3*pi/2
    
def calcul_angle_v3(pointref: Tuple[int, int], point2: Tuple[int, int]):
    dx = sqrt((pointref[0]-point2[0])**2)
    dy = sqrt((pointref[1]-point2[1])**2)
    distance = sqrt(dx**2+dy**2)

    if distance == 0:
        return float('inf')  #  éviter la division par zéro
    
    if point2[0] >= pointref[0] and point2[1] <= pointref[1] : # on est dans le quart bas a droite
        return acos(dy/distance) + pi
    
    if point2[0]>= pointref[0] and point2[1] >= pointref[1] : # quart haut a droite
        return  acos(dx/distance) +3*pi/2

    if point2[0] <= pointref[0] and point2[1] >= pointref[1] : #quart haut gauche
        return  acos(dy/distance)

    if point2[0] <= pointref[0] and point2[1] <= pointref[1] : # quart en bas a gauche
        return acos(dx/distance) + pi/2


def algo1(cloud_points: List[Tuple[int,int]]):

    # on commence par recuperer le point le plsu a gauche ppg
    ppg : Tuple[int, int]  = cloud_points[0]
    for i in range(len(cloud_points)):
        if cloud_points[i][0] <= ppg[0] : 
            ppg = cloud_points[i]

    # on recupere le point le plus a droite
    ppd : Tuple[int, int] = cloud_points[0]
    for i in range(len(cloud_points)):
        if cloud_points[i][0]>= ppd[0] :
            ppd = cloud_points[i] 
    final_points: List[Tuple[int, int]] = []
    final_points.append(ppg)

    # on execute l'algo tant que la boucle n'est pas bouclee
    # bien prendre en compte que au debut avec un seul element ca fait une boucle bouclee
    while (final_points[-1] != final_points[0] or len(final_points)==1):
        # ensuite on calcul l'ensemble des angles alpha et on cherche a le minimiser


        angles_possibles = []
        point_associe_angle = []
        for point in cloud_points:
            if (final_points[-1] != point):
                    if (ppd not in final_points):
                    #if (point not in  final_points) :
                        #angle:float = calcul_angle(final_points[-1], point)
                        angle:float = calcul_angle_v2(final_points[-1], point)
                        angles_possibles.append(angle)
                        point_associe_angle.append([point, angle])
                    # une fois ppd atteint on change de strategie
                    else :
                        angle:float = calcul_angle_v3(final_points[-1], point)
                        angles_possibles.append(angle)
                        point_associe_angle.append([point, angle])
        # on recupere le tuple associe a l'angle minimal et on renvoie le point associe : [0]
        point_associe_au_min_angle : Tuple[int, int] = min(point_associe_angle, key = lambda x:x[1])[0]
        final_points.append(point_associe_au_min_angle)

        if (len(final_points) > len(cloud_points)):
            break

    return final_points


    
# Générer des points aléatoires
def generate_random_points(n: int, x_limit: int, y_limit: int) -> List[Tuple[int, int]]:
    return [(random.randint(0, x_limit), random.randint(0, y_limit)) for _ in range(n)]

# Nombre de points et limites de l'aire
num_points = 40
x_limit = 100
y_limit = 100

# Générer des points aléatoires
cloud_points = generate_random_points(num_points, x_limit, y_limit)

# Calculer l'enveloppe convexe
envelope = algo1(cloud_points)

# Tracer les points et l'enveloppe convexe
plt.figure(figsize=(10, 8))
plt.scatter(*zip(*cloud_points), color='blue', label='Points aléatoires')
plt.scatter(*zip(*envelope), color='red', label='Enveloppe convexe', marker='o')

# Dessiner l'enveloppe convexe
for i in range(len(envelope)):
    plt.plot([envelope[i][0], envelope[(i + 1) % len(envelope)][0]], 
             [envelope[i][1], envelope[(i + 1) % len(envelope)][1]], color='red')

plt.title('Enveloppe convexe d\'un nuage de points')
plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.legend()
plt.grid()
plt.show()


#############################################################################################################

import matplotlib.pyplot as plt
import random


def orientation(p, q, r):
    # calcul determinant de matrice de rotation pour savoir si c'est à gauche ou à droite (>0=gauche, <0 = à droite)
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def enveloppe_convexe_sklansky(points):
    points = sorted(points)
    enveloppe_sup = []
    for p in points:
        while len(enveloppe_sup) >= 2 and orientation(enveloppe_sup[-2], enveloppe_sup[-1], p) <= 0:
            enveloppe_sup.pop()
        enveloppe_sup.append(p)
    enveloppe_inf = []
    for p in reversed(points):
        while len(enveloppe_inf) >= 2 and orientation(enveloppe_inf[-2], enveloppe_inf[-1], p) <= 0:
            enveloppe_inf.pop()
        enveloppe_inf.append(p)
    return enveloppe_sup[:-1] + enveloppe_inf[:-1]


def afficher_enveloppe_convexe(points, enveloppe):
    x_points, y_points = zip(*points)
    x_enveloppe, y_enveloppe = zip(*enveloppe)
    plt.scatter(x_points, y_points, label="Points", color="blue")
    plt.plot(x_enveloppe + (x_enveloppe[0],), y_enveloppe + (y_enveloppe[0],), color="red", label="Enveloppe convexe")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Enveloppe convexe avec l'algorithme de Sklansky")
    plt.legend()
    plt.show()


def generer_points(n, min_val=0, max_val=100):
    points = [(random.randint(min_val, max_val), random.randint(min_val, max_val)) for _ in range(n)]
    return points


if __name__ == "__main__":
    
    points = generer_points(50)
    enveloppe = enveloppe_convexe_sklansky(points)
    afficher_enveloppe_convexe(points, enveloppe)

######################################################################################

# Etape 1 : polygone Simple
#########################################################################################

def angle_avec_centroide(point: Tuple[int, int], centroide: Tuple[float, float]) -> float:
    return math.atan2(point[1] - centroide[1], point[0] - centroide[0])

def calculer_centroide(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return sum(x_coords) / len(points), sum(y_coords) / len(points)

def generer_polygone_simple(n: int, xmin: int = 0, xmax: int = 10, ymin: int = 0, ymax: int = 10) -> List[Tuple[int, int]]:
    points = set()
    while len(points) < n:
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        points.add((x, y))
    
    points = list(points) 
    centroide = calculer_centroide(points)
    points.sort(key=lambda p: angle_avec_centroide(p, centroide))
    
    return points



#ETape 2 : Algo de Slansky
################################################################################################################################################################################

def enveloppe_convexe_jarvis(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    n = len(points)
    if n < 3:
        return points
    
    enveloppe = []
    point_de_depart = min(points, key=lambda p: p[0]) # point le plus a gauche
    point_actuel = point_de_depart
    
    while True:
        enveloppe.append(point_actuel)
        prochain_point = points[0]
        for point in points:
            if (prochain_point == point_actuel) or (orientation(point_actuel, prochain_point, point) > 0):
                prochain_point = point
        
        point_actuel = prochain_point
        if point_actuel == point_de_depart:
            break
    
    return enveloppe


# etape 3  : Affichage
###############################################################################################################################

def afficher_polygone_et_enveloppe(points: List[Tuple[int, int]], enveloppe: List[Tuple[int, int]]):
    x_points, y_points = zip(*points)
    
    enveloppe.append(enveloppe[0])  
    x_enveloppe, y_enveloppe = zip(*enveloppe)
    
    x_polygone, y_polygone = list(x_points), list(y_points)
    x_polygone.append(x_polygone[0])
    y_polygone.append(y_polygone[0])

    plt.figure()
    plt.plot(x_polygone, y_polygone, 'b-', label="Polygone simple", linewidth=1)
    plt.plot(x_points, y_points, 'bo', label="Points du polygone")
    plt.plot(x_enveloppe, y_enveloppe, 'r-', label="Enveloppe convexe", linewidth=2)
    
    for i, point in enumerate(points):
        plt.text(point[0], point[1], f'P{i}', fontsize=12, color='blue', ha='right')
        
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygone simple et son enveloppe convexe (Sklansky)')
    plt.grid(True)
    plt.show()


points = generer_polygone_simple(20)
enveloppe = enveloppe_convexe_jarvis(points)
afficher_polygone_et_enveloppe(points, enveloppe)
        
