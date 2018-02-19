#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:30:03 2017

@author: vignesh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Limits
MINP = 1e-11
MAXP = 1 - MINP
MAXEXP = 700
MINEXP = -MAXEXP

class Feature:    
    def __init__(self, f_id, cost, f_name):
        # Feature index according to dataset
        self.f_id = f_id 
        self.cost = cost
        self.f_name = f_name


class Weight:    
    def __init__(self, stage_hash, f_id, val, trust_region):
        self.stage_hash = stage_hash
        # If it's a bias, then f_id = -1
        self.f_id = f_id
        self.val = val
        self.trust_region = trust_region

       
class Stage:    
    def __init__(self, features, f_soft_pass, f_soft_pass_1der, f_soft_pass_2der, s_name):
        self.features =  features
        # Sort in increasing order of id for binary search while populating cascade weights
        self.features.sort(key = lambda x: x.f_id)
        # Pass function
        self.f_soft_pass = f_soft_pass
        # Pass function first derivative
        self.f_soft_pass_1der = f_soft_pass_1der
        # Pass function second derivative
        self.f_soft_pass_2der = f_soft_pass_2der
        self.s_name = s_name
        self.cost = 0
        self.threshold = 0
        self.weights = [] 
    
    def sigmoid(self, x):
        x = min(max(x, MINEXP), MAXEXP)
        return 1 / (1 + np.exp(-x))

    # In this case, pass probability is NOT EQUAL to the sigmoid-probability for the given stage
    # Instead, it is defined as a separate function
    # The sigmoid-probability is used ultimately in the hard cascade as before
    def probability(self, x):
        return min(max(self.sigmoid(np.dot(self.weights, x)), MINP), MAXP)
        
    # The pass probability is used only for training the soft cascade
    def pass_probability(self, x):
        return min(max(self.f_soft_pass(np.dot(self.weights, x)), MINP), MAXP)
        
    def pass_probability_1der(self, x):
        return self.f_soft_pass_1der(np.dot(self.weights, x))
        
    def pass_probability_2der(self, x):
        return self.f_soft_pass_2der(np.dot(self.weights, x))
    
        
class Cascade:    
    def populate_cascade_weights(self, share_weights):        
        # Binary search routine
        def binsearch(lis, key):
            l = 0
            r = len(lis) - 1
            while(l <= r):
                mid = int((l + r) / 2)
                if lis[mid].f_id == key:
                    return True
                elif lis[mid].f_id > key:
                    r = mid - 1
                else:
                    l = mid + 1
            return False                           
        n_stages = len(self.stages)
        weights = []
        # Populate biases for each stage
        for i in range(n_stages):
            temp = np.zeros(n_stages)
            temp[i] = 1
            w = Weight(temp, -1, 0.01, 10)
            weights.append(w)        
        # Weight sharing among different stages
        if share_weights:
            for i in range(self.n_features):
                temp = np.zeros(n_stages)
                for j in range(n_stages):
                    if binsearch(self.stages[j].features, i):
                        temp[j] = 1
                w = Weight(temp, i, 0.01, 10)
                weights.append(w)         
        # No weight sharing among different stages
        else:
            for i in range(self.n_features):
                for j in range(n_stages):
                    if binsearch(self.stages[j].features, i):
                        temp = np.zeros(n_stages)
                        temp[j] = 1
                        w = Weight(temp, i, 0.01, 10)
                        weights.append(w)
        return weights
        
    def update_stage_weights(self):
        n_stages = len(self.stages)        
        # Reset the stage's weights
        for i in range(n_stages):
            self.stages[i].weights.clear()        
        # Populate it again
        for weight in self.weights:
            for i in range(n_stages):
                if weight.stage_hash[i]:
                    self.stages[i].weights.append(weight.val)
        
    def __init__(self, stages, n_features, share_weights):
        self.stages = stages
        # Number of features
        self.n_features = n_features
        # Populate weights of the cascade
        self.weights = self.populate_cascade_weights(share_weights)
        # Populate weights[] array of each stage in the cascade
        self.update_stage_weights()
        # Stores final thresholds
        self.thresholds = []        
        # Populate cost of each stage in the cascade
        n_stages = len(stages)
        extracted = False * np.ones(self.n_features)
        for i in range(n_stages):
            for feature in self.stages[i].features:
                if not extracted[feature.f_id]:
                    self.stages[i].cost += feature.cost
                    extracted[feature.f_id] = True
                        
    def classify(self, x):
        cost = 0
        n_stages = len(self.stages)
        for i in range(n_stages - 1):
            # Initialiize empty subset of features for the given stage
            subset_x = [1]
            # Append the features corresponding to the given stage from the feature vector
            for feature in self.stages[i].features:
                subset_x.append(x[feature.f_id])
            cost += self.stages[i].cost            
            if self.stages[i].pass_probability(subset_x) < self.stages[i].threshold:
                if self.stages[i].probability(subset_x) >= 0.5:
                    return 1, cost, i 
                else:
                    return 0, cost, i         
        # Handle last stage separately
        subset_x = [1]
        for feature in self.stages[n_stages - 1].features:
            subset_x.append(x[feature.f_id])
        cost += self.stages[n_stages - 1].cost
        if self.stages[n_stages - 1].probability(subset_x) >= 0.5:
            return 1, cost, n_stages - 1
        else:
            return 0, cost, n_stages - 1
        
    def compute_accuracy(self, X, Y, print_category):
            acc = 0
            acquisition_cost = 0
            total_stages_cost = 0
            n_examples = len(X) 
            count_correct = []
            count_wrong = []
            for stage in self.stages:
                total_stages_cost += stage.cost
            for i in range(n_examples):
                category, cost, stage_no = self.classify(X[i])
                if print_category:
                    print(category)
                acquisition_cost += cost
                if category == Y[i]:
                    acc += 1
                    count_correct.append(stage_no + 1)
                else:
                    count_wrong.append(stage_no + 1)
            return 100 * acc / n_examples, acquisition_cost / (n_examples * total_stages_cost), count_correct, count_wrong
            
    def precompute_subsets(self, X):
        subset = []
        for x in X:
            temp1 = []
            for stage in self.stages:
                temp2 = [1]
                for feature in stage.features:
                        temp2.append(x[feature.f_id])
                temp1.append(temp2)
            subset.append(temp1)
        return subset
        
    def train(self, X, Y, low_ALPHA, high_ALPHA, step_ALPHA, BETA, low_THRESH, high_THRESH, step_THRESH, PERCENT_CROSS, visualize, stats):        
        n_examples = len(X)
        n_stages = len(self.stages)
        n_weights = len(self.weights)
        n_cross = int(n_examples * PERCENT_CROSS / 100)
        n_train = n_examples - n_cross
        X_train = X[:n_train,:]
        Y_train = Y[:n_train]
        X_cross = X[n_train:,:]
        Y_cross = Y[n_train:]        
        # Precompute train and cross-validate subsets
        subset_train = self.precompute_subsets(X_train)
        subset_cross = self.precompute_subsets(X_cross)        
        #======================================================#
        # ALPHA : l1-norm coefficient                          #
        # low_ALPHA : lower limit of ALPHA for tuning          #
        # high_ALPHA : upper limit of ALPHA for tuning         #
        # step_ALPHA : step size for searching ALPHA           #
        # BETA : mean cost coefficient                         #
        # low_THRESH : lower limit for threshold selection     #
        # high_THRESH : upper limit for threshold selection    #
        # step_THRESH : step size for threshold selection      #
        #======================================================#        
        
        def compute_positive_prob(subset):
            p = []
            n = len(subset)
            for i in range(n):
                p.append(0)
                prod = 1
                for j in range(n_stages):
                    if j > 0:
                        prev_pass_prob = self.stages[j - 1].pass_probability(subset[i][j - 1])
                        # Overflow check maybe required
                        prod = prod * prev_pass_prob
                    pos_prob =  self.stages[j].probability(subset[i][j])
                    if j == n_stages - 1:
                        p[i] = min(max(p[i] + prod * pos_prob, MINP), MAXP)
                    else:
                        pass_prob = self.stages[j].pass_probability(subset[i][j])
                        p[i] = min(max(p[i] + prod * (1 - pass_prob) * pos_prob, MINP), MAXP)
            return p
                
        def objective(input_weights):
            # Assign the weights sent by minimize() to the stage weights
            for i in range(n_weights):
                self.weights[i].val = input_weights[i]
            self.update_stage_weights()            
            T = 0
            for i in range(n_train):
                expected_cost = 0
                prod = 1
                for j in range(n_stages):
                    if j > 0:
                        prev_pass_prob = self.stages[j - 1].pass_probability(subset_train[i][j - 1])
                        # Overflow check maybe required
                        prod = prod * prev_pass_prob
                    expected_cost += self.stages[j].cost * prod
                T += expected_cost                
            M1 = np.dot(Y_train, np.log(p))
            M2 = np.dot(np.ones(n_train) - Y_train, np.log(np.ones(n_train) - p))
            l = M1 + M2
            l1_norm = sum(np.abs(input_weights))
            J = -1 * l + ALPHA * l1_norm + BETA * T
            return J
                
        def objective_der(input_weights):
            # Assign the weights sent by minimize() to the stage weights
            for i in range(n_weights):
                self.weights[i].val = input_weights[i]
            self.update_stage_weights()            
            dp_dw = []
            dT_dw = []            
            for j in range(n_weights):                
                temp_dp_dw = []
                total_sum_dT_dw = 0                
                for i in range(n_train):                    
                    # Computing first and second derivatives of the probabilities
                    sum_dp_dw = 0
                    # Prefix product of pass functions (pass probabilities)
                    prod = [1]
                    # First derivative of the prefix product
                    prod_1der = [0]                    
                    for k in range(n_stages):                
                        if k > 0:
                            prev_pass_prob = self.stages[k - 1].pass_probability(subset_train[i][k - 1])
                            prev_pass_prob_1der = self.stages[k - 1].pass_probability_1der(subset_train[i][k - 1]) * self.weights[j].stage_hash[k - 1] 
                            # If the weight is not a bias
                            if self.weights[j].f_id != -1:
                                prev_pass_prob_1der *= X_train[i][self.weights[j].f_id]        
                            # Using Product Rule
                            # First derivative 
                            prod_1der.append(prod[-1] * prev_pass_prob_1der + prev_pass_prob * prod_1der[-1])
                            # Overflow check maybe required
                            prod.append(prod[-1] * prev_pass_prob)
                        pos_prob =  self.stages[k].probability(subset_train[i][k])                        
                        if k == n_stages - 1:
                            # Remaining term's first derivative
                            rem_1der = pos_prob * (1 - pos_prob) * self.weights[j].stage_hash[k]
                            # If the weight is not a bias
                            if self.weights[j].f_id != -1:
                                rem_1der *= X_train[i][self.weights[j].f_id]
                            # Product Rule 
                            # First derivative
                            term_dp_dw = prod[-1] * rem_1der + pos_prob * prod_1der[-1]                        
                        else:
                            pass_prob = self.stages[k].pass_probability(subset_train[i][k])
                            pass_prob_1der = self.stages[k].pass_probability_1der(subset_train[i][k])
                            # Remaining term's first derivative
                            rem_1der = (1 - pos_prob) * (1 - pass_prob) - pass_prob_1der
                            rem_1der *= pos_prob * self.weights[j].stage_hash[k]
                            # If the weight is not a bias
                            if self.weights[j].f_id != -1:
                                rem_1der *= X_train[i][self.weights[j].f_id]
                            # Product Rule 
                            # First derivative
                            term_dp_dw = prod[-1] * rem_1der + (1 - pass_prob) * pos_prob * prod_1der[-1]                         
                        sum_dp_dw += term_dp_dw                         
                    temp_dp_dw.append(sum_dp_dw)                    
                    # Computing first and second derivatives of the cost
                    sum_dT_dw = 0
                    for l in range(n_stages):    
                        sum_dT_dw += self.stages[l].cost * prod_1der[l] 
                    total_sum_dT_dw += sum_dT_dw                        
                dp_dw.append(temp_dp_dw)
                dT_dw.append(total_sum_dT_dw)                      
            # Conversion to numpy arrays for fast computation
            dp_dw = np.array(dp_dw)
            dT_dw = np.array(dT_dw)                
            M1 = Y_train / p 
            M2 = np.ones(n_train) - Y_train
            M3 = np.ones(n_train) - p 
            M4 = M1 - M2 / M3      
            dl_dw = np.dot(dp_dw, M4)                    
            dnorm_dw = np.sign(input_weights)
            dJ_dw = -1 * dl_dw + ALPHA * dnorm_dw + BETA * dT_dw
            return dJ_dw
                        
        #=====================================================================================================
        # THRESHOLD HELPER BEGINS HERE
        #=====================================================================================================
        # Searches for suitable thresholds through a grid search
        def threshold_helper(cur, max_acc, arg_max_cost):
            # Base case
            if cur == n_stages - 1:
                acc, cost, count_c, count_w = self.compute_accuracy(X_cross, Y_cross, False)
                if acc > max_acc:
                    # Update maximum accuracy
                    max_acc = acc
                    # Update arg-max
                    arg_max_cost = cost
                    self.thresholds.clear()
                    for i in range(n_stages):
                        self.thresholds.append(self.stages[i].threshold) 
                return max_acc, arg_max_cost
            # Recursion 
            i = low_THRESH
            while(i < high_THRESH):
                self.stages[cur].threshold = i
                max_acc, arg_max_cost = threshold_helper(cur + 1, max_acc, arg_max_cost)
                i += step_THRESH
            return max_acc, arg_max_cost
        #=====================================================================================================
        # THRESHOLD HELPER ENDS HERE
        #=====================================================================================================
        
        # Search for suitable value of ALPHA
        min_error = 1e100
        # Stores final weights i.e. arg-min
        best_weights = []        
        ALPHA = low_ALPHA
        while(ALPHA < high_ALPHA):
            p = compute_positive_prob(subset_train)
            res = minimize(objective, 100 * np.ones(n_weights), method = 'BFGS', jac = objective_der, options = {'gtol': 1e-6, 'disp': True})
            trained_weights = res.x
            for i in range(n_weights):
                self.weights[i].val = trained_weights[i]
            self.update_stage_weights()            
            p = compute_positive_prob(subset_cross)
            M1 = np.dot(Y_cross, np.log(p))
            M2 = np.dot(np.ones(n_cross) - Y_cross, np.log(np.ones(n_cross) - p))
            log_likelihood = M1 + M2            
            J_cross = -log_likelihood 
            if J_cross < min_error:
                # Update minimum error
                min_error = J_cross
                # Update arg-min
                best_weights.clear()
                for i in range(n_weights):
                    best_weights.append(self.weights[i].val)                    
            # Reset weights for the next call to train_helper 
            for i in range(n_weights):
                self.weights[i].val = 0.01 
            # Reset stage weights
            self.update_stage_weights()            
            # Display cross-validation error
            print("ALPHA = %.2f | Cross-validation error : %f" % (ALPHA, J_cross))
            print("===========================================================\n")            
            # Update ALPHA
            ALPHA *= step_ALPHA        
        # Set weights to corresponding arg-min 
        for i in range(n_weights):
            self.weights[i].val = best_weights[i]
        # Update stage weights to corresponding arg-min
        self.update_stage_weights()                       
        # Search for suitable set of thresholds
        cross_acc, cross_cost = threshold_helper(0, 0, 0)        
        # Display cross-validation accuracy
        print("Cross-validation final accuracy : %.2f %%" % cross_acc)
        print("Cross-validation final normalized-cost : %.2f" % cross_cost)        
        # Set thresholds to corresponding arg-max 
        for i in range(n_stages):
            self.stages[i].threshold = self.thresholds[i]  
    
    def test(self, X, Y):
        acc, cost, count_c, count_w = self.compute_accuracy(X, Y, False)
        return acc, cost, count_c, count_w