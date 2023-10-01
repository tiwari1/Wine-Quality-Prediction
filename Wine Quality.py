#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv('winequality-red.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[10]:


features = [feature for feature in df.columns]

fig = plt.figure(figsize=(18, 20))
for index, column in enumerate(features):
    plt.subplot(6, 2, index + 1)
    sns.histplot(df[column], kde = True)
fig.tight_layout(pad = 1.0)


# In[11]:


ax = sns.countplot(x='quality', data=df)


# In[12]:


#check for outlier
for feature in features:
    data = df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column = feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[13]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot = True)


# In[14]:


fig = plt.figure(figsize=(15,15))
for index, feature in enumerate(features):
    plt.subplot(8, 2, index + 1)
    sns.scatterplot(x= feature, y = 'quality', data = df)
fig.tight_layout(pad = 1.0)


# In[15]:


#feature engg.
df.head()


# In[16]:


df_copy = df.copy()


# In[17]:


for feature in features:
    if 0 in df_copy[feature].unique():
        pass
    else:
        df_copy[feature] = np.log(df_copy[feature])
df_copy.head()


# In[18]:


fig = plt.figure(figsize=(18, 20))

for index, feature in enumerate(features):
    plt.subplot(6, 2, index +1)
    sns.histplot(df_copy[feature], kde = True)
fig.tight_layout(pad = 1.0)


# In[19]:


print(f"Shape before: {df_copy.shape}")
df_copy = df_copy.drop(df_copy[df_copy["residual sugar"] > 5.0].index)
df_copy = df_copy.drop(df_copy[df_copy["chlorides"] > 0.2].index)
df_copy = df_copy.drop(df_copy[df_copy["sulphates"] > 1.125].index)
print(f"Shape after: {df_copy.shape}")


# In[20]:


df_copy = df_copy.drop(["density", "fixed acidity"], axis = 1)
df_copy.head()


# In[21]:


df_copy1 = df_copy.copy()


# In[22]:


df_copy['quality'] = np.exp(df_copy['quality']) 
df_copy.head()


# In[23]:


from sklearn.feature_selection import VarianceThreshold

data = df_copy.copy()
data = data.drop(["quality"], axis = 1)
var_thresh = VarianceThreshold(threshold = 0.1)
transformed_data = var_thresh.fit_transform(data)


# In[24]:


var_thresh.get_support()


# In[25]:


transformed_data


# In[26]:


selected_features_idx = []
for i, f in enumerate(var_thresh.get_support()):
    if f:
        selected_features_idx.append(i)


# In[27]:


selected_features_idx


# In[28]:


data.head()


# In[29]:


selected_features = data.columns[selected_features_idx]
selected_features


# In[30]:


df_copy.head()


# In[31]:


df_copy_selected_features = df_copy[selected_features]
df_copy_selected_features.head()


# In[32]:


y = df_copy['quality']
y


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_copy_selected_features, y, test_size = 0.3, random_state = 0)


# In[34]:


X_train.head()


# In[35]:


from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending = True)
mutual_info


# In[36]:


from sklearn.feature_selection import SelectKBest
sel_cols = SelectKBest(mutual_info_regression, k = 3) # Let's see first with 3
sel_cols.fit(X_train, y_train)
X_train.columns[sel_cols.get_support()]


# In[37]:


X_train.head()


# In[38]:


X_test.head()


# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
top_three_features = ['volatile acidity', 'free sulfur dioxide', 'total sulfur dioxide']
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# In[41]:


print("The rmse for Linear Regression is:  ", rmse)


# In[ ]:




