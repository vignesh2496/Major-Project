#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:54:32 2017

@author: vignesh
"""

import libraries as lb

color = ['r', 'g', 'b', 'y']
label = ['High Pred. Power -> High Cost, Last', 'High Pred. Power -> Low Cost, Last', 'High Pred. Power -> Low Cost, First', 'High Pred. Power -> High Cost, First']

# Loop over the 3 scenarios
for i in range(3):
    
    # Plot accuracy
    fig, ax = lb.plt.subplots()
    fig.set_size_inches(10, 6) 
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.set_xlabel(r'$ \beta $', fontsize = 16)
    ax.set_xscale('log', basex = 2)
    ax.set_ylabel('Accuracy (%)', fontsize = 16)
    # Loop over the 4 cases
    for j in range(4):
        path = 'Results/Scenario_%d/Normal/case_%d.csv' % (i + 1, j + 1)
        df = lb.pd.read_csv(path)
        df = lb.np.array(df)
        beta = df[:, 1]
        test_acc = df[:, 2]
        ax.plot(beta, test_acc, color = color[j], linestyle = '-', linewidth = 2.0, alpha = 1.0, marker = 'o', markersize = 6.0, label = label[j]) 
    ax.legend(loc = 1, ncol = 1, fancybox = True, framealpha = 0.25, fontsize = 16) 
    name = 'Results/Scenario_%d/Normal/acc.png' % (i + 1)
    fig.savefig(name, dpi = 300, bbox_inches = 'tight')  

    # Plot cost
    fig, ax = lb.plt.subplots()
    fig.set_size_inches(10, 6) 
    ax.tick_params(axis = 'both', labelsize = 16)
    ax.set_xlabel(r'$ \beta $', fontsize = 16)  
    ax.set_xscale('log', basex = 2)
    ax.set_ylabel('Normalized Mean Cost', fontsize = 16) 
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Loop over the 4 cases
    for j in range(4):
        path = "Results/Scenario_%d/Normal/case_%d.csv" % (i + 1, j + 1)
        df = lb.pd.read_csv(path)
        df = lb.np.array(df)
        beta = df[:, 1]
        cost = df[:, 3]
        ax.plot(beta, cost, color = color[j], linestyle = '-', linewidth = 2.0, alpha = 1.0, marker = '^', markersize = 6.0, label = label[j]) 
    ax.legend(loc = 1, ncol = 1, fancybox = True, framealpha = 0.25, fontsize = 16) 
    name = 'Results/Scenario_%d/Normal/cost.png' % (i + 1)
    fig.savefig(name, dpi = 300, bbox_inches = 'tight') 
