# importing modules
import numpy as np
import math
import random
# preceptron defination
class perceptron:
    def __init__(self,input_list,bias,w_list,act_func):
        self.bias = bias
        self.input_list = input_list
        self.act_func = act_func
        self.w_list = w_list
    def output(self):
        z =  np.multiply(self.input_list,self.w_list) + self.bias
        return self.act_func(z)
    def output_z(self):
        return np.multiply(self.input_list,self.w_list) + self.bias
# layer defination
class layer:
    def __init__(self,prev_layer_output,curr_layer_percep_num):
        self.curr_layer_percep_num = curr_layer_percep_num
        self.prev_layer_output = prev_layer_output
        curr_layer = []
        w_matrix = []
        for i in range(curr_layer_percep_num):
            bias = random.random()
            w_list = [random.random() for i in range(len(prev_layer_output))]
            neuron = perceptron(self.prev_layer_output,bias,w_list,sigmoid)
            curr_layer.append(neuron)
            w_matrix.append(w_list)
        self.w_matrix = w_matrix
        self.curr_layer = curr_layer
    def output_z(self):
        return np.array([percep.output_z() for percep in self.curr_layer])
    def output(self):
        return np.array([percep.output() for percep in self.curr_layer])
# functions of preceptron
# activation function
def sigmoid(x):
    return 1/(1+ math.exp(-x))
# cost function
def quad_func(output_matrix,target_matrix):
    n = len(target_matrix)
    res = np.subtract(target_matrix,output_matrix)
    res = np.array([(i**2)/(2*n) for i in res])
    return res
# output matrix z
# backpropogation function
# BP1 equation
def cost_gradient_act(output_matrix,target_matrix):
    return np.subtract(target_matrix,output_matrix)
def sigmoid_derv(z):
    return sigmoid(z)*(1-sigmoid(z))
def sigmoid_derv_matrix(output_z_matrix):
    return np.array(list(map(sigmoid_derv,output_z_matrix)))
def layer_error(output_matrix,target_matrix):
    return np.multiply(cost_gradient_act(output_matrix,target_matrix),sigmoid_derv_matrix)
# BP2 equation
def layer_error_prev(w_matrix,output_matrix,target_matrix):
    return np.multiply(np.matmul(w_matrix,layer_error(output_matrix,target_matrix)),sigmoid_derv_matrix)
# BP3 equation : FINAL BIAS GRADIENT
def cost_grad_bias(w_matrix,output_matrix,target_matrix,j):
    layer_error_matrix = layer_error_prev(w_matrix,output_matrix,target_matrix)
    return layer_error_matrix[j]
# BP4 equation : FINAL WEIGHTS GRADIENT
def cost_grad_weights(output_matrix_prev_layer,error_matrix,k,j):
    return output_matrix_prev_layer[k]*error_matrix[j]
# gradient descent
def grad_descent(layer,learning_rate,target_matrix):
    prev_cost = 1
    next_cost = 0
    while(prev_cost > next_cost):
        current = layer.curr_layer
        # obtaining output matrix
        out_list = layer.output()
        # getting prev cost function
        prev_cost = quad_func(out_list,target_matrix)
        # calculating gradients and modifying weights and biases for each perceptron
        w_matrix = layer.w_matrix
        for j in range(len(current)):
            neuron = current[j]
            grad_bias = cost_grad_bias(w_matrix,out_list,target_matrix,j)
            neuron.bias -= learning_rate*grad_bias
            for k in range(len(layer.prev_layer_percep)):
                prev_output_matrix = layer.prev_layer_output
                error_matrix = layer_error(prev_output_matrix,target_matrix)
                w = cost_grad_weights(prev_output_matrix,error_matrix,k,j)
                neuron[k] -= w
        next_cost = quad_func(out_list,target_matrix)

