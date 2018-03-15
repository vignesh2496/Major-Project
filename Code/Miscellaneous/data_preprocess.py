#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:13:13 2018

@author: vignesh
"""

import numpy as np
import pandas as pd

# Read data
data = pd.read_csv('/home/vignesh/Desktop/Major-Project/Data_Normalization/cancer_data.csv')

# Separate features from labels
features = data.iloc[:, 0:30]
labels = data.iloc[:, -1:]

# Convert to numpy
features = np.array(features, dtype = np.float64)
labels = np.array(labels)

# 0-1 Normalization
for i in range(features.shape[1] - 1):
    maximum = np.max(features[:, i])
    minimum = np.min(features[:, i])
    features[:, i] = (features[:, i] - minimum) / (maximum - minimum)
      
# Assign binary labels
labels[np.where(labels[:] == 'M')] = 1
labels[np.where(labels[:] == 'B')] = 0
       
# Convert back to dataframe and unite
features = pd.DataFrame(features)
labels = pd.DataFrame(labels)
data = pd.concat([features, labels], axis = 1)

# Save as csv
data.to_csv('/home/vignesh/Desktop/Major-Project/Data_Normalization/normalized_cancer_data.csv', sep = ',')

# Save as numpy array
data = np.array(data, dtype = np.float64)
np.save('/home/vignesh/Desktop/Major-Project/Data_Normalization/normalized_cancer_data.npy', data)