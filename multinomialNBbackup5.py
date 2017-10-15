
# coding: utf-8

# In[3]:

import pandas as pd                                                              
import re                                       
from collections import defaultdict                                                
import numpy as np                          
import csv                                                                         
import nltk  
from sklearn.model_selection import train_test_split
get_ipython().magic('matplotlib inline')
import matplotlib.mlab as mlab
import math
import scipy.stats
from sklearn.preprocessing import normalize


# In[9]:

df = pd.read_csv("C:\\Users\\shwet\\Desktop\\YouTube-Spam-Collection-v1\\Youtube01-Psy.csv")
print(df.count())
N=350


# In[49]:

train = range(int(N * 0.8))
test = range(int(N * 0.8), N)


# In[11]:

def load(filename):
    stemmer = nltk.stem.PorterStemmer()                     
    
    pattern_remove_non_aplanumeric = re.compile('\W+') 
    
    documents = []
    categories = defaultdict(int)
    words = defaultdict(int)
    
    with open(filename, encoding="ascii", errors="surrogateescape") as fid: #the dataset is not encoded with the usual utf 8 encoding and shoes unicode error
        reader = csv.reader(fid)                                 #ascii generally maps into 256 characters without loss of data 
        
        header = next(reader)                                
        
        count = 0
        for line in reader:                                                     
            category = line[4]                                                
            document = line[1].strip()                                       
            document = re.sub(pattern_remove_non_aplanumeric, ' ', document)  
            document = [str.lower(w) for w in document.split()]                
            document = [stemmer.stem(w) for w in document]                  
            document = [w for w in document if len(w) > 1]                   
            
            categories[category] += 1                                       
            for w in document:        
                words[w] += 1                                       
            
            documents.append((category, document))                            
    
    return categories, words, documents 


# In[12]:

categories, words, documents = load("C:\\Users\\shwet\\Desktop\\YouTube-Spam-Collection-v1\\Youtube01-Psy.csv")


# In[13]:

print(categories.get)


# In[14]:

for c in sorted(categories, key=categories.get, reverse=True):
    print(c)


# In[15]:


Class = dict()
c_id = 0
for c in sorted(categories, key=categories.get, reverse=True):
    Class[c] = c_id
    c_id += 1


# In[16]:

vocab = dict()
w_no = 0


# In[17]:

for w in sorted(words, key=words.get, reverse=True): 
    if words[w] < 3:
        break
    vocab[w] = w_no
    w_no += 1


# In[18]:


# Encode documents as bag of words from vocabulary
for j, d in enumerate(documents):
    cls = Class[d[0]]
    document = [vocab[w] for w in d[1] if w in vocab]
    unique, counts = np.unique(document, return_counts=True)
    documents[j] = (cls, np.asarray((unique, counts)).T)


# In[25]:

# Shuffle data at random
rnd = np.random.RandomState(seed=1923)
rnd.shuffle(documents)


# In[27]:


    def __init__(nfeatures, nclasses, reg=1.0):
        feat = np.zeros((nclasses, nfeatures))        
        coun = np.zeros(nclasses)                   
        
    def update(document, cls):
        for (feature, count) in document:
            feat[cls, feature] += count         
            coun[cls] += 1                      
    
    def predict(document):
        score = np.log(coun + 1)
        for (feature, count) in document:
            score += count * np.log(self.feat[:, feature] + self.reg) 
                   - count * np.log(np.sum(self.feat, axis=1) + self.reg*self.nfeatures)

        prob = np.exp(score - np.min(score))                 
        
        norm_prob = prob / np.sum(prob)
        return norm_prob
        
    def estimates(self):
        norm = np.sum(self.feat, axis=1) + self.reg*self.nfeatures
        return (self.feat + self.reg) / np.kron(norm, np.ones((self.nfeatures, 1))).transpose()


# In[32]:

def calc_acc(docs):
    total = 0
    right_class = 0
    for a in docs:
        total += 1
        (cls, document) = documents[a]
        probabilities = classifier.predict(document)
        prediction = np.argmax(probabilities)
        if prediction == cls:
            right_class += 1

    accuracy = float(right_class) / total    
    return accuracy


# In[50]:

classifier = MultinomialNB(nclasses=len(Class), nfeatures=len(vocab))

for j in train:
    (cls, document) = documents[j]
    #classifier.update(document, cls)
    update(document, cls)


# In[51]:

print ('accuracy={:f}'.format(calc_acc(test)))


# In[ ]:



