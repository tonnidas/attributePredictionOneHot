# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
# ----------------------------------------

# for i in range(7):

#     # Loading attributes from a pickle
#     with open(f'z_score_normalized_old_{i}.pickle', 'rb') as handle: A = pickle.load(handle)
#     with open(f'z_score_normalized_new_{i}.pickle', 'rb') as handle: B = pickle.load(handle)

#     print((A==B).all())

with open('accuracy.pickle', 'rb') as handle: A = pickle.load(handle)

print(A)