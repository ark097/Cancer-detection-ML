#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics 
from sklearn.metrics import accuracy_score


# In[8]:


df = pd.read_csv("data.csv")
df.head(20)


# In[4]:


df.drop("id",axis=1,inplace=True)


# In[5]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})


# In[8]:


features_mean= list(df.columns[1:11])
features_se= list(df.columns[11:21])
features_worst=list(df.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)


# In[6]:


df.head()


# In[159]:


sns.countplot(df['diagnosis'])


# In[160]:


plt.scatter(df['area_mean'],df['perimeter_mean'])


# In[161]:


corr = df[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'Blues') 


# In[26]:


df1=df.drop(features_worst,axis=1)


# In[27]:


cols = ['perimeter_mean','perimeter_se','area_mean','area_se']
df1 = df1.drop(cols, axis=1)


# In[28]:


cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df1 = df1.drop(cols, axis=1)


# In[29]:


df1.head()


# In[30]:


useful_features=list(df1.columns[1:])
print(useful_features)


# In[31]:


corr1=df.corr()
corr1.nlargest(20,['diagnosis'])['diagnosis']


# In[32]:


x=df1
y=df1['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(len(x_train),'\t',len(x_test))


# In[33]:


model=LogisticRegression()


# In[34]:


model.fit(x_train[useful_features],y_train)


# In[35]:


pred=model.predict(x_test[useful_features])


# In[40]:


print(model.score(x_train[useful_features],y_train)*100)


# In[172]:


accuracy=metrics.accuracy_score(pred,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))


# # 2nd try

# In[11]:


xx =df[[ 'concave points_worst', 'perimeter_worst', 
       'concave points_mean', 'radius_worst',
       'perimeter_mean', 'area_worst',
       'radius_mean', 'area_mean',
       'concavity_mean', 'concavity_worst',
       'compactness_mean', 'compactness_worst',
       'radius_se', 'area_se', 'perimeter_se']]
yy = df['diagnosis'] 


# In[12]:


xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.2, random_state=1)


# In[13]:


model2=LogisticRegression()
model2.fit(xx_train,yy_train)
print(model2.score(xx_train,yy_train)*100)


# In[14]:


pred=model2.predict(xx_test)
print(pred)
print(metrics.accuracy_score(pred,yy_test)*100)


# In[15]:


model2=LogisticRegression()
model2.fit(xx_test,yy_test)
print(model2.score(xx_test,yy_test)*100)


# # 3rd try - Random forest

# In[20]:


model3=RandomForestClassifier()
model3.fit(xx_train,yy_train)
pred=model3.predict(xx_test)
pred


# In[21]:


print(model3.score(xx_train,yy_train)*100) #trainset accuracy


# In[22]:


model3=RandomForestClassifier()
model3.fit(xx_test,yy_test)
print(model3.score(xx_test,yy_test)*100) #testset accuracy


# In[23]:


print(metrics.accuracy_score(pred,yy_test)*100)


# In[ ]:




