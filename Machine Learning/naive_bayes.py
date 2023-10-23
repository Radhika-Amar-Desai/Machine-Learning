"""
    IMPORT MODULES
"""
import numpy as np
import pandas as pd
"""
    GET THE DATA
"""
data  = pd.read_csv("play.csv")
data = data[:10]
"""
    DEFINE FUNCTIONS
"""
def conditional_prob(feature,i,dataset,target):
    tot = len(dataset[dataset[feature] == i]) + 2
    p1 = len(dataset[(dataset["play"] == target) & (dataset[feature] == i)]) + 1 #black box adding
    p2 = p1/tot
    return p2

def correct_prob(feature,dataset):      #this probability is not meant for target it is meant for parameters as it comes with black box correction
    groups = dataset[feature].unique()
    targets = dataset["play"].unique()
    dict = {}
    for i in groups:
        targets_dict = {}
        for j in targets:
            targets_dict[j] = conditional_prob(feature,i,dataset,j)
        dict[i] = targets_dict
    return dict

def target_prob(dataset):
    tot = list(dataset["play"])
    target_dict = {}
    for i in dataset["play"].unique():
        p1 = tot.count(i)
        target_dict[i] = p1/len(tot)
    return target_dict
"""
    MAIN DRIVER CODE
"""
target_dict = target_prob(data)
"""
    INPUT
"""
test_data = pd.read_csv("play.csv")
input_row = test_data.iloc[12]
"""
    PROCESSING WITH USER DEFINED FUNCTIONS
"""
output_target_dict = {}
p_final_target = {}
for i in target_dict:
    p_target = target_dict[i]
    for j in data.columns[1:len(data.columns)-1]:
        para = input_row[j]
        para_dict = correct_prob(j,data)
        p_para = para_dict[para][i]
        p_target *= p_para
    p_final_target[i] = p_target
p_max = max(p_final_target.values())
"""
    OUTPUT
"""
for i in p_final_target:
    if(p_final_target[i] == p_max):
        print(i)