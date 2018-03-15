#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:30:03 2017

@author: vignesh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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
                    
    def reset_all_weights(self):
        n_weights = len(self.weights)  
        # Reset cascade weights
        for i in range(n_weights):
            self.weights[i].val = 0.01 
        # Reset stage weights
        self.update_stage_weights()  
        
    def reset_all_thresholds(self):
        n_stages = len(self.stages)
        for i in range(n_stages):
            self.stages[i].threshold = 0
        self.thresholds.clear()
    
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
    
    def train(self, X, Y, low_ALPHA, high_ALPHA, step_ALPHA, BETA, ETA, EPSILON, ITERATIONS, DEC_PERIOD, DEC_FACTOR, low_THRESH, high_THRESH, step_THRESH, visualize, stats):        
        n_stages = len(self.stages)
        n_weights = len(self.weights)   

        #======================================================#
        # ALPHA : l1-norm coefficient                          #
        # low_ALPHA : lower limit of ALPHA for tuning          #
        # high_ALPHA : upper limit of ALPHA for tuning         #
        # step_ALPHA : step size for searching ALPHA           #
        # BETA : mean cost coefficient                         #
        # ETA : initial learning rate                          # 
        # EPSILON : convergence parameter                      #
        # ITERATIONS : maximum number of iteraions             #
        # DEC_PERIOD : period for reducing the learning rate   #
        # DEC_FACTOR : reduce factor for learning rate         #
        # low_THRESH : lower limit for threshold selection     #
        # high_THRESH : upper limit for threshold selection    #
        # step_THRESH : step size for threshold selection      #
        #======================================================#    
        
        def precompute_subsets(X):
            subset = []
            for x in X:
                temp1 = []
                for stage in self.stages:
                    temp2 = [1]
                    for feature in stage.features:
                        temp2.append(x[feature.f_id])
                    temp1.append(temp2)
                subset.append(temp1)
            return np.array(subset)
        
        def compute_positive_prob(subset):
            prob = []
            n = len(subset)
            for i in range(n):
                prob.append(0)
                prod = 1
                for j in range(n_stages):
                    if j > 0:
                        prev_pass_prob = self.stages[j - 1].pass_probability(subset[i][j - 1])
                        # Overflow check maybe required
                        prod = prod * prev_pass_prob
                    pos_prob =  self.stages[j].probability(subset[i][j])
                    if j == n_stages - 1:
                        prob[i] = min(max(prob[i] + prod * pos_prob, MINP), MAXP)
                    else:
                        pass_prob = self.stages[j].pass_probability(subset[i][j])
                        prob[i] = min(max(prob[i] + prod * (1 - pass_prob) * pos_prob, MINP), MAXP)
            return np.array(prob)
            
        # TRAIN SUB-ROUTINE STARTS HERE
        # =============================        
        def train_helper(X_train, Y_train, ALPHA, ETA):
            convergence = False            
            cycles = []
            train_error = []
            iterations = 0     
            subset_train = precompute_subsets(X_train)
            n_train = subset_train.shape[0]
            # Newton's Algorithm begins
            while not convergence and iterations < ITERATIONS:        
                # Adaptive learning rate
                if iterations % DEC_PERIOD == 0 and iterations != 0:
                    ETA /= DEC_FACTOR                
                # Increment iterations
                iterations += 1                
                # Compute initial probabilities for each example at the beginning of the cycle
                p = compute_positive_prob(subset_train)
                sum_p = sum(p)                    
                dp_dw = []
                dT_dw = []
                d2p_dw2 = [] 
                d2T_dw2 = []                
                for j in range(n_weights):                    
                    temp_dp_dw = []
                    temp_d2p_dw2 = []
                    total_sum_dT_dw = 0
                    total_sum_d2T_dw2 = 0                    
                    for i in range(n_train):                        
                        # Computing first and second derivatives of the probabilities
                        sum_dp_dw = 0
                        sum_d2p_dw2 = 0
                        # Prefix product of pass functions (pass probabilities)
                        prod = [1]
                        # First derivative of the prefix product
                        prod_1der = [0]
                        # Second derivative of the prefix product
                        prod_2der = [0]                        
                        for k in range(n_stages):                
                            if k > 0:
                                prev_pass_prob = self.stages[k - 1].pass_probability(subset_train[i][k - 1])
                                prev_pass_prob_1der = self.stages[k - 1].pass_probability_1der(subset_train[i][k - 1]) * self.weights[j].stage_hash[k - 1] 
                                prev_pass_prob_2der = self.stages[k - 1].pass_probability_2der(subset_train[i][k - 1]) * self.weights[j].stage_hash[k - 1]
                                # If the weight is not a bias
                                if self.weights[j].f_id != -1:
                                    prev_pass_prob_1der *= X_train[i][self.weights[j].f_id]
                                    prev_pass_prob_2der *= X_train[i][self.weights[j].f_id] ** 2
                                # Using Product Rule
                                # Second derivative 
                                prod_2der.append(prod[-1] * prev_pass_prob_2der + 2 * prod_1der[-1] * prev_pass_prob_1der + prev_pass_prob * prod_2der[-1] )
                                # First derivative 
                                prod_1der.append(prod[-1] * prev_pass_prob_1der + prev_pass_prob * prod_1der[-1])
                                # Overflow check maybe required
                                prod.append(prod[-1] * prev_pass_prob)
                            pos_prob =  self.stages[k].probability(subset_train[i][k])                            
                            if k == n_stages - 1:
                                # Remaining term's first and second derivatives
                                rem_1der = pos_prob * (1 - pos_prob) * self.weights[j].stage_hash[k]
                                rem_2der = pos_prob * (1 - pos_prob) * (1 - 2 * pos_prob) * self.weights[j].stage_hash[k]
                                # If the weight is not a bias
                                if self.weights[j].f_id != -1:
                                    rem_1der *= X_train[i][self.weights[j].f_id]
                                    rem_2der *= X_train[i][self.weights[j].f_id] ** 2
                                # Product Rule 
                                # First derivative
                                term_dp_dw = prod[-1] * rem_1der + pos_prob * prod_1der[-1]
                                # Second derivative
                                term_d2p_dw2 = prod[-1] * rem_2der + 2 * prod_1der[-1] * rem_1der + pos_prob * prod_2der[-1]                            
                            else:
                                pass_prob = self.stages[k].pass_probability(subset_train[i][k])
                                pass_prob_1der = self.stages[k].pass_probability_1der(subset_train[i][k])
                                pass_prob_2der = self.stages[k].pass_probability_2der(subset_train[i][k])
                                # Remaining term's first and second derivatives
                                rem_1der = (1 - pos_prob) * (1 - pass_prob) - pass_prob_1der
                                rem_2der = (1 - pos_prob) * rem_1der - ((1 - pos_prob) * pass_prob_1der + (1 - pass_prob) * pos_prob * (1 - pos_prob) + pass_prob_2der)
                                rem_1der *= pos_prob * self.weights[j].stage_hash[k]
                                rem_2der *= pos_prob * self.weights[j].stage_hash[k]
                                # If the weight is not a bias
                                if self.weights[j].f_id != -1:
                                    rem_1der *= X_train[i][self.weights[j].f_id]
                                    rem_2der *= X_train[i][self.weights[j].f_id] ** 2
                                # Product Rule 
                                # First derivative
                                term_dp_dw = prod[-1] * rem_1der + (1 - pass_prob) * pos_prob * prod_1der[-1]
                                # Second derivative
                                term_d2p_dw2 = prod[-1] * rem_2der + 2 * prod_1der[-1] * rem_1der + (1 - pass_prob) * pos_prob * prod_2der[-1]                             
                            sum_dp_dw += term_dp_dw
                            sum_d2p_dw2 += term_d2p_dw2                             
                        temp_dp_dw.append(sum_dp_dw)
                        temp_d2p_dw2.append(sum_d2p_dw2)                        
                        # Computing first and second derivatives of the cost
                        sum_dT_dw = 0
                        sum_d2T_dw2 = 0
                        for l in range(n_stages):    
                            sum_dT_dw += self.stages[l].cost * prod_1der[l] 
                            sum_d2T_dw2 += self.stages[l].cost * prod_2der[l]
                        total_sum_dT_dw += sum_dT_dw    
                        total_sum_d2T_dw2 += sum_d2T_dw2                        
                    dp_dw.append(temp_dp_dw)
                    d2p_dw2.append(temp_d2p_dw2)
                    dT_dw.append(total_sum_dT_dw)
                    d2T_dw2.append(total_sum_d2T_dw2)                       
                # Conversion to numpy arrays 
                p = np.array(p)
                dp_dw = np.array(dp_dw)
                dT_dw = np.array(dT_dw)
                d2p_dw2 = np.array(d2p_dw2) 
                d2T_dw2 = np.array(d2T_dw2)                
                M1 = Y_train / p 
                M2 = np.ones(n_train) - Y_train
                M3 = np.ones(n_train) - p 
                M4 = M1 - M2 / M3      
                dl_dw = np.dot(dp_dw, M4)                
                # Get list of weight values
                weights_val = []
                for weight in self.weights:
                    weights_val.append(weight.val)    
                dnorm_dw = np.sign(weights_val)
                dJ_dw = -1 * dl_dw + ALPHA * dnorm_dw + BETA * dT_dw                
                M5 = np.dot(d2p_dw2, M4) 
                M6 = Y_train / p ** 2 + M2 / M3 ** 2
                M7 = np.dot(dp_dw ** 2, M6)
                d2l_dw2 = M5 - M7
                d2J_dw2 = -1 * d2l_dw2 + BETA * d2T_dw2                
                # Newton-Raphson update 
                # dw = -1 * dJ_dw / d2J_dw2
                # Gradient Descent update 
                dw = -dJ_dw                
                for i in range(n_weights):
                    dw[i] = min(max(dw[i], -self.weights[i].trust_region), self.weights[i].trust_region)
                    self.weights[i].trust_region = max(2 * abs(dw[i]), self.weights[i].trust_region / 2)
                # Update self.weights
                for i in range(n_weights):
                    self.weights[i].val += ETA * dw[i]
                # Update stage weights
                self.update_stage_weights()                     
                # Compute new +ve probability for each example
                p_new = compute_positive_prob(subset_train)                                
                # Compute training-loss and soft training accuracy
                soft_acc = 0
                sum_abs_diff = 0
                log_likelihood = 0
                for i in range(n_train):
                    sum_abs_diff += abs(p_new[i] - p[i])
                    log_likelihood += Y_train[i] * np.log(p_new[i]) + (1 - Y_train[i]) * np.log(1 - p_new[i])
                    if p_new[i] >= 0.5:
                        category = 1
                    else:
                        category = 0
                    if category == Y_train[i]:
                        soft_acc += 1
                soft_acc /= n_train
                soft_acc *= 100                
                # Convergence criterion
                convergence = sum_abs_diff/sum_p <= EPSILON                
                # Compute training error (without the regularization term)
                J_train = -log_likelihood                
                if visualize:                                                          
                    # Visualize the graph of  negative of penalized log-likelihood VS no. of cycles
                    cycles.append(iterations)
                    train_error.append(J_train) 
                    plt.plot(cycles, train_error, 'ro')
                    plt.axis([0, 100, 0, 1000])
                    plt.xlabel("Cycle number")
                    plt.ylabel("Training loss")
                    plt.show()                
                if stats:
                    # Print Statistics
                    print("Epoch %d: Training loss : %f | Soft Training Accuracy : %.2f %%" % (iterations, J_train, soft_acc))           
                    print("-----------------------------------------------------------")                                    
            # Newton's Algorithm ends
        # TRAIN SUB-ROUTINE ENDS HERE
        # ===========================
        
        #=====================================================================================================
        # THRESHOLD HELPER BEGINS HERE
        #=====================================================================================================
        # Searches for suitable thresholds through a grid search
        def threshold_helper(X_cross, Y_cross, cur, metric):
            # Base case
            if cur == n_stages - 1:
                acc, cost, count_c, count_w = self.compute_accuracy(X_cross, Y_cross, False)
                if (1 - acc / 100) < metric:
                    # Update metric
                    metric = (1 - acc / 100) 
                    self.thresholds.clear()
                    for i in range(n_stages):
                        self.thresholds.append(self.stages[i].threshold) 
                return metric
            # Recursion 
            i = low_THRESH
            while(i < high_THRESH):
                self.stages[cur].threshold = i
                metric = threshold_helper(X_cross, Y_cross, cur + 1, metric)
                i += step_THRESH
            return metric
        #=====================================================================================================
        # THRESHOLD HELPER ENDS HERE
        #=====================================================================================================
        
        # Search for suitable value of ALPHA
        min_metric = np.inf       
        ALPHA = low_ALPHA
        BEST_ALPHA = 0
        while(ALPHA < high_ALPHA):
            seed = 11
            kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = seed)
            cv_metric = []
            for train, cross in kfold.split(X, Y):
                X_train = X[train]
                X_cross = X[cross]
                Y_train = Y[train]
                Y_cross = Y[cross]
                train_helper(X_train, Y_train, ALPHA, ETA)
                metric = threshold_helper(X_cross, Y_cross, 0, np.inf)
                self.reset_all_weights()
                self.reset_all_thresholds()
                cv_metric.append(metric)
            if  np.mean(cv_metric) < min_metric:
                # Update minimum metric
                min_metric = np.mean(cv_metric)
                # Update best ALPHA
                BEST_ALPHA = ALPHA        
            # Display cross-validation metric
            print("ALPHA = %.2f | Cross-validation metric : %f +- %f" % (ALPHA, min_metric, np.std(cv_metric)))
            print("===========================================================\n")            
            # Update ALPHA
            ALPHA *= step_ALPHA            
        train_helper(X, Y, BEST_ALPHA, ETA)
        threshold_helper(X, Y, 0, np.inf)             
        # Set thresholds to corresponding arg-min
        for i in range(n_stages):
            self.stages[i].threshold = self.thresholds[i]  
    
    def test(self, X, Y):
        acc, cost, count_c, count_w = self.compute_accuracy(X, Y, False)
        return acc, cost, count_c, count_w