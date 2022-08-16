# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
# ----------------------------------------

# ----------------------------------------
# Loading attributes from a pickle
with open('attributes.pickle', 'rb') as handle:
    Vertices_attributes = pickle.load(handle)

numNodes = len(Vertices_attributes)
print('num nodes:', numNodes)

# Loading edges from a pickle
with open('edges.pickle', 'rb') as handle:
    EdgesList = pickle.load(handle)

numEdges = len(EdgesList)
print('num edges:', numEdges)

# Making adjacency matrix (considering it is directed graph)
EdgesList_matrix = np.zeros((numNodes, numNodes))

for index, row in EdgesList.iterrows():
    u, v = int(row['From']), int(row['To'])
    EdgesList_matrix[u][v] = 1 

friend_row_without_zero_matrix = np.zeros((numNodes, numNodes)) # zero padded adjacency list
adjList = EdgesList.groupby('From')['To'].apply(list).to_dict()

for i in range(numNodes):
    if i in adjList:
        for j in range(len(adjList[i])):
            friend_row_without_zero_matrix[i][j] = adjList[i][j]


# ----------------------------------------

def getNHopNeighbors(node, hop): # It is simply a bfs till nhop, not on whole graph
    neighborsTillHop = set()
    n_neighbors = {node}

    for i in range(hop):
        temp = set()

        for curNode in n_neighbors:
            if curNode in adjList:
                temp = temp.union(set(adjList[curNode]))
        
        neighborsTillHop = neighborsTillHop.union(temp)
        n_neighbors = temp
    
    return neighborsTillHop
# ----------------------------------------
def error_computation(attribute1, a1, a1_distinct, n_hope_neighbors):
 
        all_nodes_a1_distribution = np.zeros((len(a1), len(a1_distinct)))
        
        # loop starts for each node

        for node in range(len(a1)): # node means "each_immediate_friend" here 
            Immediate_friends_Nodes = getNHopNeighbors(node, n_hope_neighbors) # list of adjacent nodes of "each_immediate_friend" till n_hop

            for friend in Immediate_friends_Nodes:
                friend_a1_attribute = Vertices_attributes.iat[friend, attribute1]

                if friend_a1_attribute != 0:
                    friend_a1_attribute_index = np.where(a1_distinct == friend_a1_attribute)[0]
                    all_nodes_a1_distribution[node][friend_a1_attribute_index] = all_nodes_a1_distribution[node][friend_a1_attribute_index] + 1
        
        # loop ends for each node
                  
        z_score_normalized = all_nodes_a1_distribution / all_nodes_a1_distribution.sum(axis=1)[:,None]
        z_score_normalized[np.isnan(z_score_normalized)] = 0 # replace NaN i.e. 0/0 witn 0
      
        return z_score_normalized
# ----------------------------------------

# ----------------------------------------
# Necessary variables for mixing matrix making
n_hope_neighbors = 1
attributeCount = len(Vertices_attributes.columns) # number of attributes

for attribute2 in range(attributeCount):

    print('predicting', Vertices_attributes.columns[attribute2])

    a2 = Vertices_attributes.iloc[:,attribute2]
    a2_distinct = a2.unique()
    a2_distinct = a2_distinct[a2_distinct != 0] # non-zero distinct values of a2

    zero_rows_a2 = np.where(a2.to_numpy() == 0)[0] # index of zero rows of a2

    z_score_normalized = "not_yet_initialized"

    for attribute1 in range(attributeCount):

        a1 = Vertices_attributes.iloc[:,attribute1]
        a1_distinct = a1.unique()
        a1_distinct = a1_distinct[a1_distinct != 0] # non-zero distinct values of a1
        a1_distinct = np.sort(a1_distinct)

        z_score_current = error_computation(attribute1, a1, a1_distinct, n_hope_neighbors)
        z_score_current = np.delete(z_score_current, zero_rows_a2, 0) # remove rows for which a2 value is zero # can also be done outside inner loop

        if attribute1 == 0:
            z_score_normalized = z_score_current
        else:
            z_score_normalized = np.concatenate((z_score_normalized, z_score_current), axis=1) # add new columns
    
    # inner loopends here

    # print("final z_score_normalized")
    # print(z_score_normalized)
    # print(z_score_normalized.shape)

    with open(f'z_score_normalized_old_{attribute2}.pickle', 'wb') as handle: pickle.dump(z_score_normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)
