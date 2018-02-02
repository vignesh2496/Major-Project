#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:07:44 2017

@author: vignesh
"""

import cascade as cas

'''
# Read data
df = cas.lb.pd.read_csv('Dataset/dataset4.csv')
df = cas.lb.np.array(df)
# Shuffle examples
cas.lb.shuffle(df)
'''

df = cas.lb.np.load('Dataset/dataframe1.npy')


# Make train and test sets
X_train = df[0:800, 1:11]
Y_train = df[0:800, 11] 
X_test = df[800:, 1:11]
Y_test = df[800:, 11]

# Create costs
cost = cas.lb.np.zeros(10)
for i in range(9):
    cost[i] = (i + 1) * 0.018
cost[9] = 1 - sum(cost)

# Create features
f = []
for i in range(10):
    feature = cas.Feature(i, cost[9 - i], 'f%d' % i)
    f.append(feature)

results = []
beta_list = [65]

for beta in beta_list:
    # Initialize stages
    s0 = cas.Stage([f[7], f[8], f[9]], 's0')
    s1 = cas.Stage([f[4], f[5], f[6]],'s1')
    s2 = cas.Stage([f[0], f[1], f[2], f[3]], 's2')
    
    # Initialize cascade
    c1 = cas.Cascade([s0, s1, s2], 10, False)
    
    # Relax into soft cascade and train
    # Train + Cross-validation size : 800 
    c1.train(X_train, Y_train, 0.1, 7, 2, beta, 0.1, 5e-3, 50, 1, 1, 0.1, 1.0, 0.1, 25, False, False)
    # Testing
    # Test size : 200 
    acc, cost = c1.test(X_test, Y_test)
    print("Testing accuracy : %.2f %%" % acc)
    print("Testing normalized-cost : %.2f" % cost)
    #results.append([beta, acc, cost])

#results = cas.lb.pd.DataFrame(results)
#results.to_csv('Results/Scenario_3/case_1.csv', sep=',')