# -*- coding: utf-8 -*-
"""
Created on Thu May 25 01:31:05 2017

@author: Vignesh Kannan_2496
"""

import libraries as lb

def gen_dataset_store(filename, n_samples, n_features, n_inform, n_redun, n_repeat, n_classes, weights, flip_y, scale):
    X, y = lb.datasets.make_classification(n_samples = n_samples, n_features = n_features, n_informative = n_inform, n_redundant = n_redun, n_repeated = n_repeat, n_classes = n_classes, weights = weights, flip_y = flip_y, scale = scale)
    X = lb.pd.DataFrame(X)
    y = lb.pd.DataFrame(y)
    data = lb.pd.concat([X, y], axis=1)
    data.to_csv(filename, sep=',')
    
scale = 100 * lb.np.random.rand(10)            
gen_dataset_store('Dataset/dataset4.csv', 1000, 10, 7, 0, 0, 2, None, 0.05, scale)