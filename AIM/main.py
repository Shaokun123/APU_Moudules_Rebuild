# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:48:15 2020

@author: Sanatov Daulet
"""

import os,pickle,random,operator,math
import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import mfcc
from tempfile import TemporaryFile
from collections import defaultdict

dataset = [] # intializae a empty list variable call dataset  

def loadDataset(filename):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
loadDataset("my.dat")

def distance(instance1,instance2,k):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) # use trace method of numpy to do something
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance
    
def getNeighbors(trainingSet , instance , k):
    distances =[]
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestClass(neighbors):
    classVote ={}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1 
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

#Comment next line and uncomment others if you want update dataset
results=defaultdict(int,  {1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 5: 'hiphop', 6: 'jazz', 7: 'metal', 8: 'pop', 9: 'reggae', 10: 'rock'})

i=1
for folder in os.listdir("C:\\Users\\Sana\\Desktop\\APU\\AIM\\code\\genre\\"): #If you want set new dataset, then change path to the folder of data set and comment result
  results[i]=folder
  i+=1
(rate,sig)=wav.read("C:\\Users\\Sana\\Desktop\\APU\\AIM\\code\\genres\\rock\\rock.00000.wav") # Change path to song, which you want to check on genre
mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature=(mean_matrix,covariance,0)
pred=nearestClass(getNeighbors(dataset ,feature , 5))
print(results[pred])