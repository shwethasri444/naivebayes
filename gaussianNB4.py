
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
get_ipython().magic('matplotlib inline')
import matplotlib.mlab as mlab
import math
import scipy.stats
from sklearn.preprocessing import normalize

df = pd.read_csv('C:\\Users\\shwet\\Desktop\\usage\\input\\Iris.csv')


train, test = train_test_split(df, test_size = 0.2)

Prior = {}

Count = {}
Count[1]=train["Species"][train["Species"] == 'Iris-setosa'].value_counts()
Count[2]=train["Species"][train["Species"] == 'Iris-virginica'].value_counts()
Count[3]=train["Species"][train["Species"] == 'Iris-versicolor'].value_counts()

Prior[1] = Count[1]/120 
Prior[2] = Count[2]/120 
Prior[3] = Count[3]/120 

Class = {}
mean = {}
variance = {}

Class["Iris-setosa"] = df.loc[df['Species'] == "Iris-setosa"]
Class["Iris-virginica"] = df.loc[df['Species'] == "Iris-virginica"]
Class["Iris-versicolor"] = df.loc[df['Species'] == "Iris-versicolor"]

mean[1] = Class["Iris-setosa"].mean()
mean[2] = Class["Iris-virginica"].mean()
mean[3] = Class["Iris-versicolor"].mean()

variance[1] = Class["Iris-setosa"].var()
variance[2] = Class["Iris-virginica"].var()
variance[3] = Class["Iris-versicolor"].var()

def dist_func(x,i,k):
    pdfval = scipy.stats.norm( mean[k][i], math.sqrt(variance[k][i]) ).pdf(x)
    return(pdfval)

def enum_class(x):
    return {
        'Iris-setosa': 1,
        'Iris-virginica': 2,
        'Iris-versicolor': 3,
    }.get(x, -1)  

pred_prob = {}

for index, row in train.iterrows():    
    assert (row[0] == index + 1) 
    pre_norm = {}
    
    for k in range(1,4):
        like_hood = 1
        for i in range(1,5):
            like_hood = like_hood * dist_func(row[i],i,k)
        pre_norm[k] = Prior[k]*likelyhood
        

    pred_prob[index] = normalize(np.array(list(pre_norm.values())).reshape(1,-1), norm='l1')
            

array_rep = np.zeros((3,3))

for index, row in train.iterrows():
    pred_class = pred_prob[index].argmax() 
    real_class = enum_class(row[5]) 
    array_rep[real_class-1][pred_class] += 1
    
    
print(array_rep)

posterior = {}


for index, row in test.iterrows():
    
    assert (row[0] == index + 1) 

   
    pre_norm = {}

    for k in range(1,4):
        like_hood = 1
        for i in range(1,5):
            like_hood = likelyhood * dist_func(row[i],i,k)
        pre_norm[k] = Prior[k]*like_hood

    posterior[index] = normalize(np.array(list(pre_norm.values())).reshape(1,-1), norm='l1')



array_rep = np.zeros((3,3))

Counts1={}

Counts1[1]=0
Counts1[2]=0
Counts1[3]=0

for index, row in test.iterrows():
    pred_class = posterior[index].argmax() 
    if pred_class==0 :
        Counts1[1]+=1;
    elif pred_class==1 :
        Counts1[2]+=1; 
    elif pred_class==2 :
        Counts1[3]+=1;    
    real_class = enum_class(row[5]) 
    array_rep[real_class-1][pred_class] += 1

print(array_rep)
total = test.count()[0]
err = 0

for i in range(3):
    for j in range(3):
        if(i == j):
            continue
        err = err + array_rep[i][j]
        

print("\n \n error "+str(err/total)+" \n")


print(train['Species'].value_counts())
print(test['Species'].value_counts())




# In[ ]:



