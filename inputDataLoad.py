# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
# ----------------------------------------

# ----------------------------------------
# The inputs are attributes of nodes and edges of the graph
# Loading the attribute lists
Vertices_attributes_path = "/Users/tonni/Documents/LabProjects/attributePredictionOneHot/txts/UNC28_Attributes.txt"
Vertices_attributes = pd.read_csv(Vertices_attributes_path, sep="\t", header=None)
Vertices_attributes.columns = ['Status','Gender','Major','Minor','Dorm','Graduation_Year','High_school']

# Storing attributes in a pickle
with open('pickles/attributes.pickle', 'wb') as handle:
    pickle.dump(Vertices_attributes, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Loading the edges lists
EdgesList_path = "/Users/tonni/Documents/LabProjects/attributePredictionOneHot/txts/UNC28_edgeList.txt"
EdgesList = pd.read_csv(EdgesList_path, sep="\t", header=None)
EdgesList.columns = ['From','To']
EdgesList = EdgesList - 1 # use zero based indexing for node ids

# Storing edges in a pickle
with open('pickles/edges.pickle', 'wb') as handle:
    pickle.dump(EdgesList, handle, protocol=pickle.HIGHEST_PROTOCOL)