#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


#import data
train= pd.read_csv('train.csv')
train.head(40)


# In[3]:


#Menghitung jumlah NaN pada kolom
train['Pekerjaan'].value_counts()
train['Kelas Pekerja'].value_counts()


# In[4]:


#Drop data yang tidak digunakan
train=train.drop(['id','Pendidikan','Kelas Pekerja','Pekerjaan'],axis=1)


# In[5]:


#Replace Gaji
train=train.replace({'<=7jt':0,'>7jt':1})


# In[6]:


#Dummies untuk categorical
train=pd.get_dummies(train,columns = ['Status Perkawinan','Jenis Kelamin'])


# In[7]:


x_train = train.drop(['Gaji'],axis=1)
y_train = train['Gaji']


# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# In[10]:


stdscalar=StandardScaler()
datascale=stdscalar.fit_transform(x_train)


# In[11]:


colnames=['Umur','Berat Akhir','Jmlh Tahun Pendidikan','Keuntungan Kapital','Kerugian Capital','Jam per Minggu','Status Perkawinan_Belum Pernah Menikah','Status Perkawinan_Berpisah','Status Perkawinan_Cerai','Status Perkawinan_Janda','Status Perkawinan_Menikah','Status Perkawinan_Menikah LDR','Jenis Kelamin_Laki2','Jenis Kelamin_Perempuan']
dfscale=pd.DataFrame(datascale,columns=colnames)


# In[12]:


dfscale


# In[13]:


x_train


# In[14]:


model = KNeighborsClassifier()
param_grid= {'n_neighbors':np.arange(5,50),'weights':['distance','uniform']}
gscv = GridSearchCV(model,param_grid=param_grid, scoring='roc_auc',cv=5)
gscv.fit(dfscale,y_train)


# In[15]:


gscv.best_score_


# In[18]:


gscv.best_params_


# In[23]:


#Import data test
test = pd.read_csv('test.csv')
test.head(20)


# In[1]:


#Membuang kolom yang tidak dibutuhkan
test=test.drop(['id','Pendidikan','Kelas Pekerja','Pekerjaan'],axis=1)


# In[26]:


#membuat dummies
test=pd.get_dummies(test,columns = ['Status Perkawinan','Jenis Kelamin'])


# In[27]:


#membuat x_test
x_test = test[:]


# In[28]:


model_knn = KNeighborsClassifier(n_neighbors=41, weights='uniform')


# In[29]:


#Model Training
model_knn.fit(dfscale,y_train)


# In[30]:


testscale=stdscalar.fit_transform(x_test)


# In[31]:


dftestscale=pd.DataFrame(testscale,columns=colnames)


# In[32]:


#Model Testing
y_test=model_knn.predict(dftestscale)


# In[33]:


y_test


# In[34]:


#Menambahkan kolom gaji kepada data frame test
test['Gaji']=y_test


# In[35]:


id_test = np.arange(35994,45593)


# In[36]:


test['id']=id_test


# In[37]:


test


# In[38]:


Hasil = test[['id','Gaji']]


# In[39]:


Hasil.to_csv('Rafi Naufal Al-Mochtari Pohan_new.csv',index=False)


# In[ ]:



