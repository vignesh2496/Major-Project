# -*- coding: utf-8 -*-
"""
Created on Thu May 25 01:31:05 2017

@author: Vignesh Kannan_2496
"""

import numpy as np
import pandas as pd
from sklearn import datasets

def gen_dataset_store(filename, n_samples, n_features, n_inform, n_redun, n_repeat, n_classes, weights, flip_y, scale):
    X, Y = datasets.make_classification(n_samples = n_samples, n_features = n_features, n_informative = n_inform, n_redundant = n_redun, n_repeated = n_repeat, n_classes = n_classes, weights = weights, flip_y = flip_y, scale = scale)
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    data = pd.concat([X, Y], axis = 1)
    data.to_csv(filename, sep = ',')
    
scale = np.random.rand(10)            
gen_dataset_store('/home/vignesh/Desktop/Major-Project/Dataset/dataset6.csv', 1000, 10, 7, 0, 0, 2, None, 0.05, scale)