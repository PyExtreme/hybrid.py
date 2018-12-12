"""

Hybridisation of Fuzzy-C means clustering using Natural Computing Algorithms(PSO, GSA, ACO)

This is a util module containing functions used by other modules

Ankit Jha
May 2017

"""


import random as rd
from numpy import *
from numpy.random import random, uniform
from collections import Counter
from scipy.spatial import distance
from math import exp
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import math
import seaborn as sns
import random
import matplotlib.pyplot as plt
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
import decimal
from multiprocessing import Process , Manager
import operator


def ImportData(filename, delim):
    """
    This function imports the data from a file.
    """
    arr = []
    try:
        f = open(str(filename), 'r')
        for line in f:
            temp = line.split(delim)
            for j in range(0,len(temp)):
                temp[j] = int(temp[j])
            arr.append(temp)
        print("Imported Data")
        f.close()
        return arr
    except:
        raise Exception("File not found")

def ShuffleData(arr):
    """
    This function shuffles the data
    """
    iterator = range(0,len(arr))
    random.shuffle(iterator)
    res = [[] for i in range(0,len(arr))]
    for index in range(0,len(iterator)):
        res[index] = arr[iterator[index]]
    return res, iterator

def arrange(arr, iterator):
    """
    This function re-arranges the data in order before the shuffling
    """
    res = [[]for i in range(0,len(arr))]
    for index in range(len(iterator)):
        res[iterator[index]] = arr[index]
    return res

def display(arr):
    for i in range(len(arr)):
        print(arr[0])

def norm(arr, center):
    """
    This function calculates euclidean distance between vectors
    """
    if len(arr) != len(center):
        raise Exception('Input size not equal')
    temp = 0.0
    for i in range(0,len(arr)):
        temp += abs(arr[i] - center[i]) ** 2
    return math.sqrt(temp)

def InitMembershipMatrix(arr, num_clusters):
    """
    Initialise the membership matrix
    """
    return np.random.random((len(arr), num_clusters))

def DeFuzzify(U):
    """
    De-fuzzifies data
    """
    for i in range(0,len(U)):
        mx = max(U[i])
        for j in range(0,len(U[0])):
            if U[i][j] != mx:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U

def accuracy(u, data, Cluster, number_cluster):
    """
    Calculates the accuracy of the experiment
    """   
    ClusterTest = u.argmax(axis=1)
    C1 = np.asarray(Cluster)
    C2 = np.asarray(ClusterTest)
    ctp = np.count_nonzero((C1[:, None] == C1) * (C2[:, None] == C2))
    ctn = np.count_nonzero((C1[:, None] != C1) * (C2[:, None] != C2))
    cfp = np.count_nonzero((C1[:, None] == C1) * (C2[:, None] != C2))
    cfn = np.count_nonzero((C1[:, None] != C1) * (C2[:, None] == C2))
    prec = (ctp * 100 ) / (ctp + cfp)
    rec = (ctp * 100) / (ctp + cfn)
    ri = ((ctp + ctn) * 100.0) / (ctp + ctn + cfp + cfn)
    print('{}'.format(prec) + " is the precision")
    print('{}'.format(rec) + " is the recall")
    print('{}'.format(ri) + " is the RI")
    ARI = adjusted_rand_score(Cluster, ClusterTest)
    #print('{}'.format(ARI)  + " is the ARI")
    fscore = (2.0 * prec * rec) / (prec + rec)
    fscore = (f1_score(C1, C2, average = 'weighted'))
    purity = v_measure_score(C1, C2)                                                  
    ClusterPurity = [[0] for i in range(number_cluster)]
    NumClusterPurity = [[0 for i in range(number_cluster)] for j in range(number_cluster)]
    for i in range(0, len(data)):
        ClusterPurity[ClusterTest[i]].append(i)
    purity = 0.0
    for i in range(0, number_cluster):
        for j in range(0, len(ClusterPurity[i])):
            NumClusterPurity[i][Cluster[ClusterPurity[i][j]] - 1] += 1
        purity += max(NumClusterPurity[i])
    purity = float(purity) / float(len(data))
    #print('{}'.format(purity) + " is the purity")
    return ARI, fscore, purity
