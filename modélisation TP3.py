#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:00:53 2024

@author: thomasroujou
"""




from random import *

import numpy as np

from math import gamma
from math import log

from math import exp

from math import tan

#from math import abs

from math import pi


import matplotlib.animation as animation

from math import sqrt

import matplotlib.pyplot as plt

def histo1(n):
    L=[]
    for i in range(n):
        L.append(random())
    return(plt.hist(L,bins=90))

def Beta(a,n):
    L=[]
    for i in range(n):
        L.append(random()**(1/a))
    return(plt.hist(L,bins=100))

def Beta2(b,n):
    L=[]
    for i in range(n):
        L.append(random()**(1/b)+1)
    return(plt.hist(L,bins=100))

def expO(lamb,n):
    L=[]
    for i in range(n):
        L.append(log(1-random())/lamb)
    return(plt.hist(L,bins=100))
    
def pareto(a,b,n):
    L=[]
    for i in range(n):
        L.append(((1-random())**a)/(b))
    return(plt.hist(L,bins=90))

def weibull(a,b,n):
    L=[]
    for i in range(n):
        L.append((log(1-random())**(1/a)/(b)))
    return(plt.hist(L,bins=90))

def cauchy(a,b,n):
    L=[]
    for i in range(n):
        
        L.append(tan(pi*random()-pi/2)*(b)+a)
    return(plt.hist(L,bins=90))


def poisson(lamb,n):
    L=[]
    for i in range(n):
        
        L.append(tan(pi*random()-pi/2)*(b)+a)
    return(plt.hist(L,bins=90))

def Laplace(n):
    L=[]
    for i in range(n):
        a=random()
        if a<1/2:
            L.append(log(2*a))
        else:
            L.append(-log(2-2*a))
    return L[0]

def Laplacehist(n):
    L=[]
    for i in range(n):
        a=random()
        if a<1/2:
            L.append(log(2*a))
        else:
            L.append(-log(2-2*a))
    return(plt.hist(L,bins=90))

def Laplace2(x):
    return((1/2)*exp(-(abs(x))))

def Normale(x):
    return(1/(sqrt(2*pi))*exp(-(x**2)/2))

def LoiNormaleRejet(m):
    M=sqrt(2*1.718281828/pi)
    L=[]
    for i in range(m):
        y=Laplace(1)
        u=random()
        while(u>(Normale(y)/(M*Laplace2(y)))):
            y=Laplace(1)
            u=random()
        L.append(y)
    return(plt.hist(L,bins=90))

def expo(lamb,n):
    L=[]
    for i in range(n):
        L.append(-log(random())/lamb)
    return L[0]

def Loiexp(a,x):
    return(a*exp(-a*x))

def LoiGamma(a,x):
    return(a*(x**(a-1))*exp(-x)*1/(gamma(a)))

def LoiGammaRejet(m,a):
    M=sqrt(2*1.718281828/pi)
    L=[]
    for i in range(m):
        y=expo(a,1)
        u=random()
        while(u>LoiGamma(a,y)/(M*Loiexp(a,y))):
            y=expo(a,1)
            u=random()
        L.append(y)
    return(plt.hist(L,bins=90))

def pimontecarlo(n):
    a=0
    b=0
    for i in range(n):
        a=a+1
        u=random()
        v=random()
        if ((u**2+v**2)<=1):
            b=b+1
    return(4*(b/a),b)

def intervalledeconfiance(n,piestime,b):
    pestime=b/n
    #erreurtype = 4 * np.sqrt(pestime * (1 - pestime) / n)
    erreurtype2= 4 *np.sqrt((pi/4)*(1-pi/4)/n)
    z = 1.96  # pour 95% de confiance quantile 5% loi normale
   # intervalleconfiance = (piestime - z * erreurtype, piestime + z * erreurtype)
    intervalleconfiance2 = (piestime - z * erreurtype2, piestime + z * erreurtype2)
    return (intervalleconfiance2)
    #return (intervalleconfiance, intervalleconfiance2)

def graphepi(n):
    L1=[pimontecarlo(i)[0] for i in range(10,n)]
    Lsup=[intervalledeconfiance(i,pimontecarlo(i)[0],pimontecarlo(i)[1])[1] for i in range(10,n)]
    Linf=[intervalledeconfiance(i,pimontecarlo(i)[0],pimontecarlo(i)[1])[0] for i in range(10,n)]
    N=[i for i in range(10,n)]
    Pi=[pi for i in range(10,n)]
    plt.plot(N, L1, label='pi_estimé', marker='',linewidth= 0.5)
    plt.plot(N, Lsup, label='supconf', marker='',linewidth=0.5)
    plt.plot(N, Linf, label='infconf', marker='',linewidth=0.5)
    plt.plot(N, Pi, label='Pi', marker='',linewidth=1)

N=(1.96**2)*4*pi*(1-pi/4)/(0.005**2)


def pimontecarlo2(n):
    a=0
    b=0
    for i in range(n):
        u=random()
        a=a+sqrt(1-u**2)/n
    return a*4

def graphepi2(n):
    L1=[pimontecarlo2(i) for i in range(10,n)]
    Lsup=[intervalledeconfiance2(i,pimontecarlo2(i)) for i in range(10,n)]
    Linf=[intervalledeconfiance2(i,pimontecarlo2(i)) for i in range(10,n)]
    N=[i for i in range(10,n)]
    Pi=[pi for i in range(10,n)]
    plt.plot(N, L1, label='pi_estimé', marker='',linewidth= 0.5)
    plt.plot(N, Lsup, label='supconf', marker='',linewidth=0.5)
    plt.plot(N, Linf, label='infconf', marker='',linewidth=0.5)
    plt.plot(N, Pi, label='Pi', marker='',linewidth=1)
    
def intervalledeconfiance2(n,p):
    PI=pimontecarlo2(n)
    if ((2/3-PI**2/16)/n)>=0:
        erreurtype2= 4 *np.sqrt((2/3-PI)**2/16)/n
        z = 1.96  # pour 95% de confiance quantile 5% loi normale avec le pi estimé
        intervalleconfiance2 = (p - z * erreurtype2, p + z * erreurtype2)
    else:
       erreurtype2= 4 *np.sqrt((2/3-pi)**2/16)/n
       z = 1.96  # pour 95% de confiance quantile 5% loi normale avec le vrai PI en cas de pb
       intervalleconfiance2 = (p - z * erreurtype2, p + z * erreurtype2)
    
    return (intervalleconfiance2)
    
    #return (intervalleconfiance, intervalleconfiance2)
#a la place de pi pour l'intervalle de confiance on met l'stimateur donc on fait la fonction intervalle de confiance et 
#et pimontecarlo en  même temps pour pas changer l'estimateur