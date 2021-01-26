# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:39:03 2020

@author: ael-k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 
from random import shuffle 
from tqdm import tqdm 
from PIL import Image
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics

##### ignore warnings
warnings.filterwarnings('ignore')

##### define paths
print(os.listdir("SMILEs"))
current_path = os.path.dirname(os.path.abspath(__file__))
current_path = current_path.replace("\\","/")
path_smiles = current_path+"/SMILEs/positives/positives7/"
path_frowns = current_path+"/SMILEs/negatives/negatives7/"

##### show one normalized image
path = path_smiles+'6.jpg'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('test1',img)
norm_img = np.zeros((16,16))
img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('test2',img)
cv2.imwrite('norm_img.jpg', img)

##### store images in lists
smile_data = []
frown_data = []
image_size = 64
image_resize = 16

for image in tqdm(os.listdir(path_smiles)): 
    path = os.path.join(path_smiles, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ##### resize image for a faster training of the logistic regression model
    img = cv2.resize(img, (image_resize, image_resize))
    ##### normalize the images
    norm_img = np.zeros((image_resize,image_resize))
    img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
    smile_data.append(img)
    
for image in tqdm(os.listdir(path_frowns)): 
    path = os.path.join(path_frowns, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ##### resize image for a faster training of the logistic regression model
    img = cv2.resize(img, (image_resize, image_resize))
    ##### normalize the images
    norm_img = np.zeros((image_resize,image_resize))
    img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
    frown_data.append(img)

##### label data
smile_label = np.ones(len(smile_data),dtype=int)
frown_label = np.zeros(len(frown_data),dtype=int)

##### concatenate data
data = np.concatenate((np.asarray(smile_data),np.asarray(frown_data)),axis=0)
label = np.concatenate((np.asarray(smile_label),np.asarray(frown_label)),axis=0).reshape(data.shape[0],1)

##### split train and test set
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=42)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]).T
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]).T
y_test = y_test.T
y_train = y_train.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    output = 1/(1+np.exp(-z))
    return output

def forward_backward_propagation(w,b,input_data,true_output):
    ##### define number of inputs
    total_images = input_data.shape[1]
    ##### forward propagation
    z = np.dot(w.T,input_data) + b
    output = sigmoid(z)
    loss = -true_output*np.log(output)-(1-true_output)*np.log(1-output)
    cost = (np.sum(loss))/total_images
    ##### backward propagation
    derivative_weight = (np.dot(input_data,((output-true_output).T)))/total_images
    derivative_bias = np.sum(output-true_output)/total_images
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

def update(w,b,input_data,true_output,learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    for i in range(number_of_iterarion):
        cost,gradients = forward_backward_propagation(w,b,input_data,true_output)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,input_data):
    z = sigmoid(np.dot(w.T,input_data)+b)
    predicted_labels = np.zeros((1,input_data.shape[1]))
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            predicted_labels[0,i] = 0
        else:
            predicted_labels[0,i] = 1
    return predicted_labels

##### train logistic regression
learning_rate = 0.000001
num_iterations = 2001
dimension =  x_train.shape[0]
w,b = initialize_weights_and_bias(dimension)
optimalparameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

##### test logistic regression
test_predicted_labels = predict(optimalparameters["weight"],optimalparameters["bias"],x_test)
train_predicted_labels = predict(optimalparameters["weight"],optimalparameters["bias"],x_train)
accuracy = metrics.accuracy_score(y_test, test_predicted_labels.astype(int))
#confusion_matrix = metrics.confusion_matrix(y_test, test_predicted_labels.astype(int))
#classification_report = metrics.classification_report(y_test, test_predicted_labels.astype(int))

##### print results
print("Test Accuracy: {} %\n".format(round(100 - np.mean(np.abs(test_predicted_labels - y_test)) * 100,2)))
print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(train_predicted_labels - y_train)) * 100,2)))

print("accuracy:", accuracy)
print("number of training images:", x_train.shape[1])
print("number of test images:", x_test.shape[1])
#print('\nClassification Report:\n', classification_report)

# =============================================================================
# ##### using logistic regression from scikit learn
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]},
# logistic_regression=LogisticRegression(random_state=42)
# log_reg_cv=GridSearchCV(logistic_regression,grid,cv=10)
# log_reg_cv.fit(x_train.T,y_train.T)
# print("best hyperparameters: ", log_reg_cv.best_params_)
# print("accuracy: ", log_reg_cv.best_score_)
# log_reg= LogisticRegression(C=log_reg_cv.best_params_['C'],penalty=log_reg_cv.best_params_['penalty'])
# log_reg.fit(x_train.T,y_train.T)
# print("test accuracy: {} ".format(log_reg.fit(x_test.T, y_test.T).score(x_test.T, y_test.T)))
# print("train accuracy: {} ".format(log_reg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# =============================================================================
