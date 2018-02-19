#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:39:45 2017

@author: vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

color = ['r', 'g', 'b']
label = ['Normal', 'Exp', 'Min']

# Loop over the 3 scenarios
for i in range(1):
    
    # Plot accuracy
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6) 
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.set_xlabel(r'$ \beta $', fontsize = 16)
    ax.set_xscale('log', basex = 2)
    ax.set_ylabel('Accuracy (%)', fontsize = 16)
    path = []
    for j in range(3):
        path.append('Results/Scenario_%d/' % (i + 1) + label[j] + '/case_1.csv') 
        df = pd.read_csv(path[j])
        df = np.array(df)
        beta = df[:, 1]
        test_acc = df[:, 2]
        ax.plot(beta, test_acc, color = color[j], linestyle = '-', linewidth = 2.0, alpha = 1.0, marker = 'o', markersize = 6.0, label = label[j]) 
    ax.legend(loc = 1, ncol = 1, fancybox = True, framealpha = 0.50, fontsize = 16) 
    name = 'Results/Scenario_%d/compare_acc.png' % (i + 1)
    fig.savefig(name, dpi = 300, bbox_inches = 'tight')  
    
    # Plot cost
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6) 
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.set_xlabel(r'$ \beta $', fontsize = 16)
    ax.set_xscale('log', basex = 2)
    ax.set_ylabel('Normalized Mean Cost', fontsize = 16) 
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for j in range(3):
        df = pd.read_csv(path[j])
        df = np.array(df)
        beta = df[:, 1]
        cost = df[:, 3]
        ax.plot(beta, cost, color = color[j], linestyle = '-', linewidth = 2.0, alpha = 1.0, marker = '^', markersize = 6.0, label = label[j]) 
    ax.legend(loc = 1, ncol = 1, fancybox = True, framealpha = 0.50, fontsize = 16) 
    name = 'Results/Scenario_%d/compare_cost.png' % (i + 1)
    fig.savefig(name, dpi = 300, bbox_inches = 'tight') 
