# ----------------------------------------
# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
# ----------------------------------------

# ----------------------------------------
def getNHopNeighbors(node, hop, adjList): # It is simply a bfs till nhop, not on whole graph
    neighborsTillHop, n_neighbors = set(), {node}

    for i in range(hop):
        temp = set()

        for curNode in n_neighbors:
            if curNode in adjList:
                temp = temp.union(set(adjList[curNode]))
        
        neighborsTillHop = neighborsTillHop.union(temp)
        n_neighbors = temp
    
    return neighborsTillHop
# ----------------------------------------

# ----------------------------------------
def OneHotEncoding(Vertices_attributes):
    attributes = Vertices_attributes.replace(0, np.nan) # get_dummies ignores nan but does not ignore zero, we want to ignore 0 columns
    oneHot = pd.get_dummies(attributes, columns = attributes.columns)

    distinctLabelCount = dict()
    for col in Vertices_attributes.columns:
        # count prefix match
        count = 0
        for one_hot_col in oneHot.columns:
            if one_hot_col.startswith(col):
                count = count + 1
        distinctLabelCount[col] = count

    return oneHot, distinctLabelCount
# ----------------------------------------

# ----------------------------------------
def error_computation(numNodes, n_hop_neighbors, Vertices_attributes_oneHot, Vertices_attributes, adjList, distinctLabelCount):
    all_nodes_distribution = np.zeros((numNodes, len(Vertices_attributes_oneHot.columns)))

    for eachNode in range(numNodes):
        Immediate_friends_Nodes = getNHopNeighbors(eachNode, n_hop_neighbors, adjList) # gets a list of adjacent nodes till a hop; this method is written on allMethods.py
        Vertices_attributes_sum = Vertices_attributes_oneHot.iloc[list(Immediate_friends_Nodes)].sum()
        all_nodes_distribution[eachNode] = Vertices_attributes_sum.to_numpy()
    
    i = 0
    for col in Vertices_attributes.columns:
        count = distinctLabelCount[col]
        denominator = np.sum(all_nodes_distribution[ : , i : i+count], axis = 1)
        denominator = denominator[None].T
        all_nodes_distribution[ : , i : i+count] = all_nodes_distribution[ : , i : i+count] / denominator

        i = i + count
    
    all_nodes_distribution[np.isnan(all_nodes_distribution)] = 0 # replace NaN i.e. 0/0 witn 0

    return all_nodes_distribution
# ----------------------------------------

# ----------------------------------------
def accuracyMeasurement(Test_True_Labels, predicted_labels):
    matched = 0
    for i in range(len(Test_True_Labels)):
        if Test_True_Labels[i] == predicted_labels[i]:
            matched = matched + 1
    accuracy = (matched / len(Test_True_Labels)) * 100

    # confusion_matrix = dict()
    # for i in range(len(Test_True_Labels)):
    #     actual = Test_True_Labels[i] 
    #     predicted = predicted_labels[i]
    #     if (actual, predicted) not in confusion_matrix:
    #         confusion_matrix[(actual, predicted)] = 0
    #     confusion_matrix[(actual, predicted)] = confusion_matrix[(actual, predicted)] + 1

    # total, matched = 0, 0
    # for each_key in confusion_matrix:
    #     total = total + confusion_matrix[each_key]
    #     if each_key[0] == each_key[1]: matched = matched + 1

    # accuracy = (matched / total) * 100
    return accuracy
# ----------------------------------------

# ----------------------------------------
def knnClassifier(Train, Test, Train_True_Labels):
    knn = KNeighborsClassifier(n_neighbors = 10) # Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 10
    knn.fit(Train, Train_True_Labels)
    predicted_labels = knn.predict(Test)
    return predicted_labels
# ----------------------------------------

# ----------------------------------------
def svmClassifier(Train, Test, Train_True_Labels):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(Train, Train_True_Labels)
    predicted_labels = clf.predict(Test)
    return predicted_labels
# ----------------------------------------

# ----------------------------------------
def naiveBiasClassifier(Train, Test, Train_True_Labels):
    gnb = GaussianNB()
    predicted_labels = gnb.fit(Train, Train_True_Labels).predict(Test) # gnb.fit(X_train, y_train).predict(X_test)
    return predicted_labels
# ----------------------------------------

# ----------------------------------------
def decisionTreeClassifier(Train, Test, Train_True_Labels):
    dlf = DecisionTreeClassifier(random_state=48)
    predicted_labels = dlf.fit(Train, Train_True_Labels).predict(Test) # gnb.fit(X_train, y_train).predict(X_test)
    return predicted_labels
# ----------------------------------------