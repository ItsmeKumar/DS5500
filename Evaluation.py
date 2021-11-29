import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2
import math
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_similarity_score


def calculate_iou(component1,component2):
    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR

    IOU = overlap.sum()/float(union.sum())
    
    return(IOU)
    
def dice(component1,component2):
    eps = 1e-6
    pred = np.asarray(component1).astype(np.bool)
    gtruth = np.asarray(component2).astype(np.bool)

    if not np.sum(gtruth) + np.sum(pred):
        return 1.0

    return ((2. * np.sum(gtruth * pred)) /(np.sum(gtruth) + np.sum(pred) + eps))
                
def PA(component1,component2):
    eps = 1e-6
    pred = np.asarray(component1).astype(np.bool)
    gtruth = np.asarray(component2).astype(np.bool)

    if not np.sum(gtruth) + np.sum(pred):
        return 1.0

    return ((2. * np.sum(gtruth * pred)) /(np.sum(gtruth) + np.sum(pred) + eps))

def calculate_hausdroff(component1,component2):
    number_points = np.shape(component1)[0]
    factor = 0
    
    for i in range(0,number_points):
        factor = factor + max(directed_hausdorff(u, v)[i], directed_hausdorff(v, u)[i])
        
    factor = factor/number_points
    return(factor)
 
 def trimap(component1,component2):
    intermediate_result = []
    for row in range(0, 5):
        for column in range(0,5):
            intermediate_result = intermediate_result + [jaccard_similarity_score(component1[row][column],component2[row][column])]  
            
    return(np.mean(intermediate_result))

def bfScore(component1,component2):
    intermediate_result = -1
    theta = 10
    
    for i in range(0,3):
      a = component1[i]
      b = component2[i]
      if (b > a)&(abs(b-a) >= theta):
        component2[i] = component2[i] - theta
      elif(b > a)&(abs(b-a) < theta):
        component2[i] = component2[i]
      elif (b < a)&(abs(b-a) >= theta):
        component2[i] = component2[i] + theta
      elif(b < a)&(abs(b-a) < theta):
        component2[i] = component2[i]
    
    return(jaccard_similarity_score(component1,component2))

pred = pd.read_pickle(r'C:\Users\Inspiron\Desktop\DS5500\dataset')
data = pd.read_pickle(r'C:\Users\Inspiron\Desktop\DS5500\predicted')

moddata = data*256
moddata = moddata.astype(int)

actual_moddata = actual*256
actual_moddata = actual_moddata.astype(int)


# Calculaing IoU Scores
IoU_Score = []
for target in range(0,300):
    print(target)
    a = actual_moddata[target]
    b = moddata[target]
    temp_score = []
    for i in range(0,256):
        for j in range(0,256):
            temp_score = temp_score + [jaccard_similarity_score(a[i][j],b[i][j])]
    IoU_Score = IoU_Score + [np.mean(temp_score)]    
    
    
# Calculating Dice Scores
Dice_Score = []
for target in range(0,300):
    print(target)
    a = actual_moddata[target]
    b = moddata[target]
    temp_score = []
    for i in range(0,256):
        for j in range(0,256):
            temp_score = temp_score + [dice(a[i][j],b[i][j])]
    Dice_Score = Dice_Score + [np.mean(temp_score)]   
    
# Calculating Trimap
tmScore = []
for target in range(0,300):
    print(target)
    a = actual_moddata[target]
    b = moddata[target]
    temp_score = []
    for i in range(3,256):
        # for j in range(0,256):
        sub_a = a[i-2:i+3]
        sub_b = b[i-2:i+3]
        intermediate_result = []
        for row in range(0, np.shape(sub_a)[0]):
            for column in range(0,np.shape(sub_a)[1]):
              intermediate_result = intermediate_result + [jaccard_similarity_score(sub_a[row][column],sub_b[row][column])]  
    tmScore = tmScore + [np.mean(intermediate_result)]
    
    
# Calculating Boundary Metric (BF1)
for target in range(0, 300):
    a = actual_moddata[target]
    b = moddata[target]
    bf_Scores = []
    for i in range(0,np.shape(a)[0]):
        for j in range(0,np.shape(a)[1]):
            bf_Scores = bf_Scores + [bfScore(a,b)]

