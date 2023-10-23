"""
    IMPORT MODULES
"""
import numpy as np
import pandas as pd
"""
    GET DATA
"""
data = pd.read_csv("Iris.xls")
"""
    DEFINE CLASSES
"""
class decision_node:
    def __init__(self,dataset,left_node,right_node,depth,parameter,compare_val):
        self.dataset = dataset
        self.depth = depth
        self.left_node = left_node
        self.right_node = right_node
        self.parameter = parameter
        self.compare_val = compare_val
        self.label = "decision"
class leaf_node:
    def __init__(self,depth,value):
        self.value = value 
        self.depth = depth
        self.label = "leaf"
"""
    DEFINE FUNCTIONS FOR GETTING NODES
"""
def ginni_index(dataset):
    prob_class = list(dataset["Species"].unique())
    if len(prob_class) == 1:
        return 0
    total = list(dataset["Species"])
    sum = 0
    for i in prob_class:
        pi = total.count(i)/len(total)
        sum += pi**2
    ginni_index = 1-sum
    return ginni_index

def info_gain(parent_node,child_node_left,child_node_right):
    p_child_node_left  = len(child_node_left)/len(parent_node) 
    p_child_node_right = len(child_node_right)/len(parent_node)
    info_gain = ginni_index(parent_node) - p_child_node_left*ginni_index(child_node_left) -p_child_node_right*ginni_index(child_node_right)
    return info_gain

def best_split(parameter,parent_node,depth_of_parent_node):
    # getting the datasets of left and right node with max info_gain
    dataset = parent_node.dataset
    tests = sorted(dataset[parameter].unique())
    max = 0
    node_test_list = []
    for i in tests:
        left_node_dataset = dataset[(dataset[parameter] > i)]
        right_node_dataset = dataset[(dataset[parameter] <= i)]
        info_gain_data = info_gain(dataset,left_node_dataset,right_node_dataset)
        test_val = i
        node_test = [left_node_dataset,right_node_dataset,info_gain_data,test_val]
        node_test_list.append(node_test)
        if(info_gain_data > max):
            max = info_gain_data
    
    for i in node_test_list:
        if i[2] == max:
            left_node_dataset  = i[0]
            right_node_dataset = i[1]     
            compare_val = i[3]

    # storing the datasets in left and right nodes and labelling them as leaf/decision
    if(ginni_index(left_node_dataset) == 0):
        best_split_left_node = leaf_node(depth_of_parent_node+1,left_node_dataset["Species"].unique())
    else:
        best_split_left_node = decision_node(left_node_dataset,[],[],depth_of_parent_node+1,None,None)
    if(ginni_index(right_node_dataset) == 0):
        best_split_right_node = leaf_node(depth_of_parent_node+1,right_node_dataset["Species"].unique())
    else:
        best_split_right_node = decision_node(right_node_dataset,[],[],depth_of_parent_node+1,None,None)
    
    # storing the best split nodes as nodes of parent and setting up conditional for the parent node
    parent_node.left_node = best_split_left_node
    parent_node.right_node = best_split_right_node
    parent_node.parameter = parameter
    parent_node.compare_val = compare_val

"""
    MAIN DRIVER CODE
"""
# split data into training and testing 
rows = len(data)
train_rows = int(0.7*rows)
train_data = data[:train_rows]
test_data = data[train_rows:]

# making whole dataset as root and spliting into 2 child nodes
tree_dataset = train_data
root_node = decision_node(tree_dataset,[],[],0,None,None)
depth_of_decision_tree = 3

"""
    CONSTRUCTING DECISION TREE
"""
def recur(parent_node,i):
    if i > depth_of_decision_tree:
        return
    else:
        parameter = str(data.columns[1+i])
        best_split(parameter,parent_node,i)
        if(parent_node.left_node.label == "decision"):
            recur(parent_node.left_node,i+1)
        if(parent_node.right_node.label == "decision"):
            recur(parent_node.right_node,i+1)
recur(root_node,0)
"""
    TESTING
"""
# passing data through the decision tree
def test_condition(parameter,compare_val,row):
    if row[parameter] >= compare_val:
        return "left"
    else:
        return "right"

def testing_decision(current_node,row):
    
    if current_node.label == "leaf":
        global result
        result = current_node.value[0]
    else:
        test_res = test_condition(current_node.parameter,current_node.compare_val,row)
        if (test_res == "left"):
            testing_decision(current_node.left_node,row)
        else:
            testing_decision(current_node.right_node,row)

# testing for accuracy
accuracy_score = 0
for i in range(len(test_data)): 
    test_data_row = test_data.iloc[i] 
    testing_decision(root_node,test_data_row)
    if(test_data_row["Species"] == result):
        accuracy_score += 1

print(round(accuracy_score/len(test_data)*100),"%")