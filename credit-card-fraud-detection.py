#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# In[1]:


get_ipython().system('python --version')


# In[2]:


import numpy as np 
import pandas as pd
import os


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Importing Data

# In[4]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[5]:


df.info()


# In[6]:


df.Class.value_counts()


# In[7]:


df['Class'] = df['Class'].str.strip("'")
df['Class'] = df['Class'].astype(int)
df.Class


# In[8]:


cols =df[["Amount","Time","Class"]]
cols.describe().T


# # Data Visualisation

# In[9]:


corr = df.corr()
corr.style.background_gradient()


# In[10]:


sns.countplot(data = df, x='Class')


# In[11]:


class_0 = df.loc[df['Class'] == 0]["Time"]
class_1 = df.loc[df['Class'] == 1]["Time"]

sns.kdeplot(data=class_0, label="Not Fraud Transactions", shade=False)
sns.kdeplot(data=class_1, label="Fraud Transactions", shade=False)
plt.xlabel("Time[s]")
plt.ylabel("Density")
plt.title("Transactions")
plt.legend()
plt.show()


# In[12]:


var = df.columns.values
i = 0
t0 = df.loc[df['Class'] == 0]
t1 = df.loc[df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(6,5,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();


# # Analysis

# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score


# In[14]:


X = df.drop('Class', axis=1)
y = df.Class

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(random_state=40),
}


# In[16]:


print('Result of different ML Models are:')
print(' ')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    
    print(f'{name} Metrics:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print('-x' * 15)
    print(' ')


# Therefore all these models(Logistic Regression, Desicion Tree and Random forest) indentifies the the fraud transaction accurately. 
