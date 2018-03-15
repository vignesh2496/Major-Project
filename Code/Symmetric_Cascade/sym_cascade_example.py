#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:07:44 2017

@author: vignesh
"""

# Scenario 1
# Case 1 

from sym_cascade import Feature, Stage, Cascade
import numpy as np
import pandas as pd
from random import shuffle

# Limits
MAXEXP = 700
MINEXP = -MAXEXP
'''
# Read data
df = pd.read_csv('/home/vignesh/Desktop/Major-Project/Dataset/dataset6.csv')
df = np.array(df)
# Shuffle examples
#shuffle(df)
'''

df = np.load('/home/vignesh/Desktop/Major-Project/Dataset/dataframe1.npy')


# Make train and test sets
X_train = df[0:800, 1:11]
Y_train = df[0:800, 11] 
X_test = df[800:, 1:11]
Y_test = df[800:, 11]
print(np.size(np.where(Y_test==1)))
# Create costs
cost = np.zeros(10)
for i in range(9):
    cost[i] = (i + 1) * 0.018
cost[9] = 1 - sum(cost)

# Create features
f = []
for i in range(10):
    feature = Feature(i, cost[9 - i], 'f%d' % i)
    f.append(feature)

results = []
beta_list = [0]

'''
def pass_f(x):
    x = min(max(x, MINEXP), MAXEXP)
    return min(1 / (1 + np.exp(-x)), 1 - 1 / (1 + np.exp(-x)))

def pass_df(x):
    x = min(max(x, MINEXP), MAXEXP)
    sigma = 1 / (1 + np.exp(-x))  
    if x == 0:
        return 0
    elif x < 0:
        return sigma * (1 - sigma)
    else:
        return -sigma * (1 - sigma)
    
def pass_d2f(x):
    x = min(max(x, MINEXP), MAXEXP)
    sigma = 1 / (1 + np.exp(-x))  
    if x == 0:
        return 0
    elif x < 0:
        return sigma * (1 - sigma) * (1 - sigma) - sigma * sigma * (1 - sigma)
    else:
        return -(sigma * (1 - sigma) * (1 - sigma) - sigma * sigma * (1 - sigma))
'''

def pass_f(x):
    x = min(max(x, MINEXP), MAXEXP)
    return np.exp(-(x * x))

def pass_df(x):
    x = min(max(x, MINEXP), MAXEXP)
    sigma = np.exp(-(x * x))
    return -sigma * 2 * x
    
def pass_d2f(x):
    x = min(max(x, MINEXP), MAXEXP)
    sigma = np.exp(-(x * x))
    return 4 * sigma * x * x - 2 * sigma


for beta in beta_list:
    # Initialize stages
    s0 = Stage([f[7], f[8], f[9]], pass_f, pass_df, pass_d2f, 's0')
    s1 = Stage([f[4], f[5], f[6]], pass_f, pass_df, pass_d2f, 's1')
    s2 = Stage([f[0], f[1], f[2], f[3]], pass_f, pass_df, pass_d2f, 's2')
    
    # Initialize cascade
    c1 = Cascade([s0, s1, s2], 10, False)
    
    # Relax into soft cascade and train
    # Train + Cross-validation size : 800 
    c1.train(X_train, Y_train, low_ALPHA = 0.1, high_ALPHA = 0.2, step_ALPHA = 2, BETA = beta, ETA = 0.1, EPSILON = 5e-3, ITERATIONS = 50, DEC_PERIOD = 50, DEC_FACTOR = 1, low_THRESH = 0.1, high_THRESH = 1.0, step_THRESH = 0.1, PERCENT_CROSS = 25, visualize = False, stats = False)
    # Testing
    # Test size : 200 
    acc, cost, count_c, count_w = c1.test(X_test, Y_test)
    print("Testing accuracy : %.2f %%" % acc)
    print("Testing normalized-cost : %.2f" % cost)
    #results.append([beta, acc, cost])
    
    '''
    # Plot histogram of Frequency of number of stages used for classifying
    fig, ax = cas.lb.plt.subplots()
    temp = []
    temp.append(count_c)
    temp.append(count_w)
    ax.set_xticks([1,2,3])
    yticks = []
    for i in range(10, 210, 10):
        yticks.append(i)
    ax.set_yticks(yticks)
    ax.hist(temp, stacked = True, color = ['g', 'r'], label = ['Correctly Classified' ,'Wrongly Classified'], alpha = 0.75)
    ax.legend(loc = 1, ncol = 1, fancybox = True, framealpha = 0.50, fontsize = 12) 
    name = 'Results/Scenario_1/Exp/beta_6.png'
    fig.savefig(name, dpi = 300, bbox_inches = 'tight')
    '''

#results = cas.lb.pd.DataFrame(results)
#results.to_csv('Results/Min/Scenario_1/case_1_min.csv', sep=',')