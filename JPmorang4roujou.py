#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:22:10 2024

@author: thomasroujou
"""
import pandas as pd
import numpy as np


data = pd.DataFrame({
    'fico_score': [450, 500, 550, 600, 650, 700, 750, 800, 850],
    'default': [1, 1, 0, 1, 0, 0, 0, 0, 0]
})


#real data
data = pd.read_csv("/Users/thomasroujou/Desktop/Task 3 and 4_loan_Data.csv")


data = data[["fico_score", "default"]].dropna()  



    

data['bucket']= pd.cut(data['fico_score'], bins=5, labels=range(1, 5 + 1), include_lowest=True)


def calculate_mse(scores):
   #Calcule l'erreur quadratique moyenne pour un ensemble de scores.
    mean_score = scores.mean()
    mse = ((scores - mean_score) ** 2).sum()
    return mse

bucket_stats = data.groupby('bucket').agg(
    total_records=('default', 'count'),
    total_defaults=('default', 'sum'),
    mean_fico=('fico_score', 'mean'),
    mse_fico=('fico_score', calculate_mse)
)


bucket_stats['default_rate'] = bucket_stats['total_defaults'] / bucket_stats['total_records']
bucket_stats['log_likelihood'] = bucket_stats.apply(
    lambda row: row['total_defaults'] * np.log(row['default_rate'] + 1e-9) +
                (row['total_records'] - row['total_defaults']) * np.log(1 - row['default_rate'] + 1e-9),
    axis=1
)

print(bucket_stats)
