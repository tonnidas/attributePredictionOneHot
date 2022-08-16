# ----------------------------------------
# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# ----------------------------------------
from helperMethods import *
# ----------------------------------------

# ----------------------------------------
# Loading attributes from a pickle
with open('pickles/attributes.pickle', 'rb') as handle: Vertices_attributes = pickle.load(handle)

numNodes = len(Vertices_attributes)
print('number of nodes:', numNodes)

# Loading edges from a pickle
with open('pickles/edges.pickle', 'rb') as handle: EdgesList = pickle.load(handle)

numEdges = len(EdgesList)
print('number of edges:', numEdges)

# Making adjacency matrix (considering it is directed graph)
EdgesList_matrix = np.zeros((numNodes, numNodes))
adjList = EdgesList.groupby('From')['To'].apply(list).to_dict()
# ----------------------------------------

# ----------------------------------------
# one hot encoding of the attributes
Vertices_attributes_oneHot, distinctLabelCount = OneHotEncoding(Vertices_attributes)
# ----------------------------------------

# ----------------------------------------
n_hop_neighbors = 1
# ----------------------------------------

# ----------------------------------------
# calculate z_score_normalized for all attributes
print("Calculating z_score_normalized for all attributes")
z_score_normalized_all = error_computation(numNodes, n_hop_neighbors, Vertices_attributes_oneHot, Vertices_attributes, adjList, distinctLabelCount)
# ----------------------------------------

# define weights for ['Status','Gender','Major','Minor','Dorm','Graduation_Year','High_school']
# write it in a method called loadWeights
weights = dict()
weights['Status'] = np.array([5.8,0.06,0.51,0.89,0.41,3.76,1.88]) / 100
weights['Gender'] = np.array([0.06,0.22,0.14,0.27,0.28,0.09,0.52]) / 100
weights['Major'] = np.array([0.51,0.14,1.03,1.92,2.2,1.03,4.8]) / 100
weights['Minor'] = np.array([0.89,0.27,1.92,4.3,6.03,2.57,13.33]) / 100
weights['Dorm'] = np.array([0.41,0.28,2.2,6.03,22.12,1.44,13.3]) / 100
weights['Graduation_Year'] = np.array([3.76,0.09,1.03,2.57,1.44,9.37,5.53]) / 100
weights['High_school'] = np.array([1.88,0.52,4.8,13.33,13.3,5.53,27.31]) / 100

# ----------------------------------------
# storing the accuracy for each attribute and for each classifier
# knn_accuracy, svm_accuracy, naive_accuracy, dt_accuracy = [] * 7, list(), list(), list()
accuracyDf = [[0] * 7, [0] * 7, [0] * 7, [0] * 7]

# loop for each attribute
attributeCount = len(Vertices_attributes.columns) # number of attributes = 7
for attribute2 in range(attributeCount):

    print('predicting', Vertices_attributes.columns[attribute2])

    a2 = Vertices_attributes.iloc[:,attribute2]
    zero_rows_a2 = np.where(a2.to_numpy() == 0)[0] # index of zero rows of a2
    z_score_normalized = np.delete(z_score_normalized_all, zero_rows_a2, 0)
    # print(z_score_normalized, z_score_normalized.shape)

    with open(f'pickles/z_score_normalized_new_{attribute2}.pickle', 'wb') as handle: pickle.dump(z_score_normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # ----------------------------------------

    # calculate weighted z_score_normalized
    weighted_z_score_normalized = z_score_normalized
    i = 0
    for col in Vertices_attributes.columns:
        count = distinctLabelCount[col]
        weighted_z_score_normalized[ : , i : i+count] = z_score_normalized[ : , i : i+count] * weights[col][attribute2]
    
    # ----------------------------------------
    # Dropping all rows where "attribute2" columns value is 0
    Vertices_attributes2 = Vertices_attributes.drop(zero_rows_a2)

    # train test split by index
    index = range(len(weighted_z_score_normalized))
    train_ind, test_ind = train_test_split(index, test_size = 0.3, random_state = 42) # Train test set splitting as 70 30
    
    Train = weighted_z_score_normalized[train_ind, : ]
    Test = weighted_z_score_normalized[test_ind, : ]
    Train_True_Labels = Vertices_attributes2.iloc[train_ind, attribute2].to_numpy()
    Test_True_Labels = Vertices_attributes2.iloc[test_ind, attribute2].to_numpy()
    # ----------------------------------------

    # ----------------------------------------
    # Accuracy finding by knn method
    predicted_labels_knn = knnClassifier(Train, Test, Train_True_Labels)
    # print("knn predicted labels: ", predicted_labels_knn, "True labels: ", Test_True_Labels)
    accuracy_knn = accuracyMeasurement(Test_True_Labels, predicted_labels_knn)
    # print("knn accuracy:", accuracy_knn)
    accuracyDf[0][attribute2] = accuracy_knn
    # ----------------------------------------

    # ----------------------------------------
    # Accuracy finding by svm method
    predicted_labels_svm = svmClassifier(Train, Test, Train_True_Labels)
    # print("svm predicted labels: ", predicted_labels_svm, "True labels: ", Test_True_Labels)
    accuracy_svm = accuracyMeasurement(Test_True_Labels, predicted_labels_svm)
    # print("svm accuracy", accuracy_svm)
    accuracyDf[1][attribute2] = accuracy_svm
    # ----------------------------------------

    # ----------------------------------------
    # Accuracy finding by naive bias method
    predicted_labels_naive = naiveBiasClassifier(Train, Test, Train_True_Labels)
    # print("naive bias predicted labels: ", predicted_labels_naive, "True labels: ", Test_True_Labels)
    accuracy_naive = accuracyMeasurement(Test_True_Labels, predicted_labels_naive)
    # print("naive bias accuracy", accuracy_naive)
    accuracyDf[2][attribute2] = accuracy_naive
    # ----------------------------------------

    # ----------------------------------------
    # Accuracy finding by dicision tree method
    predicted_labels_dt = decisionTreeClassifier(Train, Test, Train_True_Labels)
    # print("decision tree predicted labels: ", predicted_labels_naive, "True labels: ", Test_True_Labels)
    accuracy_dt = accuracyMeasurement(Test_True_Labels, predicted_labels_dt)
    # print("decision tree accuracy", accuracy_dt)
    accuracyDf[3][attribute2] = accuracy_dt
    # ----------------------------------------

    print("True labels = ", set(Test_True_Labels))
    print("knn predicted labels = ", set(predicted_labels_knn), "svm predicted labels = ", set(predicted_labels_svm), "naive predicted labels = ", set(predicted_labels_naive), "predicted_labels_dt", set(predicted_labels_dt))
    print("knn accuracy:", accuracy_knn, "svm accuracy", accuracy_svm, "naive bias accuracy", accuracy_naive, "decision tree accuracy", accuracy_dt)
    print("# ----------------------------------------")
    # break

accuracyDf = pd.DataFrame(accuracyDf, columns=['Status','Gender','Major','Minor','Dorm','Graduation_Year','High_school'])
print(accuracyDf)
with open('pickles/accuracy.pickle', 'wb') as handle: pickle.dump(accuracyDf, handle, protocol=pickle.HIGHEST_PROTOCOL)
