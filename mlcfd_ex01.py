#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/geonhong/mlcfd/blob/master/mlcfd_ex01.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Regression ##

# ## Import Data ##

# In[1]:


import requests
import shutil

import numpy as np

# Load data from url
# Copy the file content from url to local tmp.npy file 
# then load the numpy data and return
def load_from_url(url):
  resp = requests.get(url, stream=True)
  
  with open('tmp.npy', 'wb') as f:
    shutil.copyfileobj(resp.raw, f)
   
  var = np.load('tmp.npy')
  
  return var

datin = load_from_url('https://github.com/geonhong/mlcfd/blob/master/volfrac/samples/volfrac_data.npy?raw=true')
target = load_from_url('https://github.com/geonhong/mlcfd/blob/master/volfrac/samples/volfrac_target.npy?raw=true')

print(datin.shape)
print(target.shape)

# ## Build model ##

# In[2]:


from keras import models
from keras import layers

# Build a model
def build_model():
  model = models.Sequential()
  model.add(layers.Dense(16, activation='relu', input_shape=(datin.shape[1],)))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model

model = build_model()

# Load a diamond volume fraction and predict the result

# ### Manipulate data input

# In[3]:


dataset = datin
targets = target

print(dataset.shape)
print(targets.shape)

# Shuffle data and generate train/test data set
# 80% of dataset is used to train the model and
# the rest 20% is used to test
index = np.arange(len(dataset))
np.random.shuffle(index)

train_data = []
train_targ = []

test_data = []
test_targ = []

i = 0
ntrain = 0.8*len(dataset)

for itrg in index:
  if i<ntrain:
    train_data.append(dataset[itrg])
    train_targ.append(targets[itrg])
  else:
    test_data.append(dataset[itrg])
    test_targ.append(targets[itrg])

  i += 1

train_data = np.array(train_data)
train_targ = np.array(train_targ)

test_data = np.array(test_data)
test_targ = np.array(test_targ)
    
print("train data shape: ", train_data.shape)
print("train target shape: ", train_targ.shape)
print("test data shape: ", test_data.shape)
print("test target shape: ", test_targ.shape)

# ### Fitting and evaluate the model 
# 
# 

# In[4]:


# Fit the model
num_epochs = 400

history = model.fit(train_data, train_targ, epochs=num_epochs, batch_size=1)

val_mse, val_mae = model.evaluate(test_data, test_targ)

print("MSE: ", val_mse)
print("MAE: ", val_mae)



# In[5]:


# Evaluate the model fitting

import pandas as pd

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# In[6]:


import matplotlib.pyplot as plt

def plot_hist(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure(figsize=(12,8))
  
  plt.xlabel('Epoch', fontsize=20)
  plt.ylabel('Mean Abs Error', fontsize=20)
  plt.plot(hist['epoch'], hist['mean_absolute_error'])
  
  plt.rc('xtick', labelsize=16)
  plt.rc('ytick', labelsize=16)
  
  plt.xlim([0,num_epochs])
  plt.ylim([0,1])
  
  plt.show()

plot_hist(history)

# In[7]:


# Evaluate the model
i = 0

predicted = model.predict(test_data)

cfd_data = []
cd_pred = []

for e in test_targ:
  cfd_data.append(float(e))

for e in predicted:
  cd_pred.append(float(e))
  
diffList = []
errList = []

sum_err2 = 0.0
sum_dif2 = 0.0
count = 0

def sqr(s):
  return s*s

for i in range(0, len(cd_pred)):
  diff = float(test_targ[i]) - float(cd_pred[i])
  err = diff/float(test_targ[i])
  
  diffList.append(diff)
  errList.append(err)
  
  print(i, test_targ[i], cd_pred[i], diff, err)
  
  sum_err2 += sqr(err)
  sum_dif2 += sqr(diff)
  count += 1

def mag(li):
  lo = li
  for i in range(0, len(li)):
    lo[i] = np.sqrt(li[i]*li[i])
   
  return lo

rms_err = np.sqrt(sum_err2/count)
rms_dif = np.sqrt(sum_dif2/count)

print('-----\n')
print('Diff. min/max/rms', np.min(mag(diffList)), '/', np.max(mag(diffList)), '/', rms_dif)
print('Err.  min/max/rms', np.min(mag(errList)), '/', np.max(mag(errList)), '/', rms_err)

# In[16]:




def plot_eval(x, y):
  plt.rcParams["figure.figsize"] = (12, 20)
  #plt.subplots_adjust(wspace=0.3)
  
  plt.subplot(2,1,1)
  plt.scatter(x, y, c='r')
  plt.plot([0,3], [0,3], 'b')
  plt.plot([0,3], [0,2.85], 'b--')
  plt.plot([0,3], [0,3.15], 'b--')
  
  plt.xlabel('CD of CFD', fontsize=20)
  plt.ylabel('CD of ML', fontsize=20)
  
  plt.xlim([0, 3])
  plt.ylim([0, 3])
  
  plt.subplot(2,1,2)
  plt.bar(range(len(errList)), errList)
  plt.xlabel('Case', fontsize=20)
  plt.ylabel('Error', fontsize=20)
  plt.ylim([0,0.3])
  
  plt.show()

plot_eval(cfd_data, cd_pred)
