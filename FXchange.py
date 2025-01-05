#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:21:27 2024

@author: thomasroujou
"""

# Importation des bibliothèques
import numpy as np
from scipy.stats import norm

# Modèle Garman-Kohlhagen (Options FX)
def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return price

# Exemple pour le modèle Garman-Kohlhagen
S = 1.2  # Taux spot
K = 1.25  # Strike
T = 1  # 1 an
r_d = 0.03  # Taux domestique
r_f = 0.02  # Taux étranger
sigma = 0.15  # Volatilé
print("Prix option FX (call):", garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type="call"))

# Pricing Forward FX (Parité d'intérêts)
def forward_fx(S, r_d, r_f, T):
    return S * np.exp((r_d - r_f) * T)

# Exemple pour le Forward FX
S = 1.2
r_d = 0.03
r_f = 0.02
T = 1
print("Taux Forward FX:", forward_fx(S, r_d, r_f, T))

# Swap de devises (FX Swap)
def fx_swap(S, r_d, r_f, T):
    return forward_fx(S, r_d, r_f, T)

# Exemple pour le Swap de devises
S = 1.2
r_d = 0.03
r_f = 0.02
T = 1
print("Prix FX Swap:", fx_swap(S, r_d, r_f, T))

# Modèle Black (Caps/Floors)
def black_model(L, K, T, sigma, P, option_type="call"):
    d1 = (np.log(L / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = P * (L * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        price = P * (K * norm.cdf(-d2) - L * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return price

# Exemple pour le modèle Black
L = 0.03  # Taux forward
K = 0.025  # Strike
T = 1  # 1 an
sigma = 0.2  # Volatilé
P = 1  # Facteur d'actualisation
print("Prix Caplet (call):", black_model(L, K, T, sigma, P, option_type="call"))

# Swap de Taux d'Intérêt (IRS)
def interest_rate_swap(L, K, T, P, delta):
    cash_flows = [(L - K) * delta[i] * P[i] for i in range(len(T))]
    return sum(cash_flows)

# Exemple pour le Swap de taux d'intérêt
L = 0.03
K = 0.025
T = [1, 2, 3]
P = [0.95, 0.90, 0.85]
delta = [1, 1, 1]
print("Prix IRS:", interest_rate_swap(L, K, T, P, delta))

# Monte Carlo pour les Options
def monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations=10000, option_type="call"):
    np.random.seed(42)
    dt = T
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

# Exemple pour Monte Carlo
S = 100
K = 105
T = 1
r = 0.05
sigma = 0.2
print("Prix option via Monte Carlo:", monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations=10000, option_type="call"))

# Modèle Brace-Gatarek-Musiela (LIBOR Market Model)
def bgm_model(L, sigma, T, P):
    dt = T[1] - T[0]
    drift = [sigma[i] ** 2 * dt / (1 + L[i] * dt) for i in range(len(T) - 1)]
    forward_rates = [L[i] + drift[i] for i in range(len(drift))]
    return forward_rates

# Exemple pour le modèle BGM
L = [0.03, 0.032, 0.034]
sigma = [0.2, 0.25, 0.3]
T = [1, 2, 3]
P = [0.95, 0.90, 0.85]
print("Taux forward ajustés (BGM):", bgm_model(L, sigma, T, P))

# Différences Finies (EDP pour Options)
def finite_difference_bs(S, K, T, r, sigma, N, M, option_type="call"):
    S_max = 2 * K  # Étendue maximale des prix
    dS = S_max / N
    dt = T / M
    
    # Initialisation de la grille
    grid = np.zeros((N + 1, M + 1))
    stock_prices = np.linspace(0, S_max, N + 1)
    
    # Conditions aux bords
    if option_type == "call":
        grid[:, -1] = np.maximum(stock_prices - K, 0)  # Payoff à maturité
        grid[-1, :] = S_max - K * np.exp(-r * dt * np.arange(M + 1))  # Bord supérieur
    elif option_type == "put":
        grid[:, -1] = np.maximum(K - stock_prices, 0)
        grid[0, :] = K * np.exp(-r * dt * np.arange(M + 1))  # Bord inférieur

    # Backward induction
    for j in range(M - 1, -1, -1):
        for i in range(1, N):
            delta = (grid[i + 1, j + 1] - grid[i - 1, j + 1]) / (2 * dS)
            gamma = (grid[i + 1, j + 1] - 2 * grid[i, j + 1] + grid[i - 1, j + 1]) / (dS ** 2)
            theta = -0.5 * sigma ** 2 * stock_prices[i] ** 2 * gamma - r * stock_prices[i] * delta + r * grid[i, j + 1]
            grid[i, j] = grid[i, j + 1] + dt * theta
    
    # Interpolation pour le prix initial
    S_index = int(S / dS)
    return grid[S_index, 0]

# Exemple pour Différences Finies corrigé
S = 100
K = 105
T = 1
r = 0.05
sigma = 0.2
N = 100
M = 100
print("Prix option via Différences Finies (corrigé):", finite_difference_bs(S, K, T, r, sigma, N, M, option_type="call"))