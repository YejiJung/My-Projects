#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels as sms
import scipy.stats 
import math
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
#


# In[3]:


df = pd.read_csv('~/Downloads/creditcard.csv')
df.head()


# In[4]:


# Unique value of the target value
print(f"Unique values of target variable :- \n{df['Class'].unique()}")


# In[5]:


# Plotting the Target Distribution
ax = sns.countplot(x="Class",hue="Class", data=df)
df["Class"].value_counts().transpose()
plt.title('Target Distribution',fontsize = 15)


# In[6]:


# Dataset Size
display(df.shape)


# In[7]:


# Remove irrevlevant columns
df1 = df.drop(['Time'], axis = 1)
print(f"list of feature names agter removing Time Column :- \n{df1.columns}")


# In[8]:


# Summary of data
df.describe()


# In[9]:


# Histogram graphs
f, axes = plt.subplots(8,4, figsize=(20,20))
for ax, feature in zip(axes.flat, df1.columns):
    sns.histplot(df1[feature],ax=ax)


# In[10]:


# HeatMap
plt.figure(figsize=(10,10)) 
hm = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount','Class']]
corMat = hm.corr(method='pearson')
sns.heatmap(corMat,square=True,cmap="Blues")
plt.title('Heatmap of All the Variables', fontsize=20)
plt.show()


# In[11]:


# Time distribution in hours and with Transaction Amount
plt.subplot(2, 2, 1)
(df['Time']/3600).hist(figsize=(15,15), color = "steelblue", bins = 20)
plt.title("Distribution of Time")
plt.xlabel("Hour")
plt.ylabel("Frequency")

# Transaction amount distribution by hours
plt.subplot(2, 2, 2)
plt.scatter(x = df['Time']/3600, y = df['Amount'], alpha = .8, color = "red")
plt.title("Distribution of Transaction Amount by Time")
plt.xlabel("Hour")
plt.ylabel("Frequency")
plt.show()


# In[12]:


# Summary of Fraud & Normal Transaction 
print("Fraud")
print(df.Time[df.Class == 1].describe())
print()
print("Normal")
print(df.Time[df.Class == 0].describe())


# In[13]:


# Fraud vs. Noraml in Transaction
g, (ax1, ax2)=plt.subplots(2,1,sharex=True, figsize=(12,4))

bins=50

ax1.hist(df.Time[df.Class == 1], bins=bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins=bins)
ax2.set_title('Normal')

plt.xlabel('Time (in sec)')
plt.ylabel('number of Transactions')
plt.show()


# In[14]:


# Modeling process 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Make the X and Y inputs
data_x = df.drop('Class', axis = 1)
data_y = df['Class']

# Split the data into training and testing, use 30% data to evaluate the models 
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.3, random_state = 123)

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

print("Training data has {} rows and {} variables".format(train_x.shape[0], train_x.shape[1]))
print("Testing data has {} rows and {} variables".format(test_x.shape[0], test_x.shape[1]))


# In[15]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

dummy = DummyClassifier(strategy="stratified")
dummy.fit(train_x, train_y)
dummy_pred = dummy.predict(test_x)

# Print test outcome 
print(confusion_matrix(test_y, dummy_pred))
print('\n')
print(classification_report(test_y, dummy_pred))


# In[16]:


#Build the Logistic Regression Classifier model 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 100)
logreg.fit(train_x, train_y)

pd.DataFrame(logreg.coef_, columns = df.drop('Class', axis=1).columns)
pred_y = logreg.predict(test_x)


# In[17]:


# Print the confusion matrix for the model 
conf_matrix = confusion_matrix(test_y, pred_y)

plt.figure(figsize = (7,7))
sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.title("Accuracy Score: {}".format(accuracy_score(test_y, pred_y)))
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()


# In[18]:


# Since the data is highly imbalanced, it is not enough to purly rely on the accuracy score to evaluate the model. Recall and Prescision here plays more important roles
print("The recall score for prediction is {:0.2f}".format(recall_score(test_y, pred_y, pos_label=1)))
print("The prescision score for predion is {:0.2f}".format(precision_score(test_y, pred_y, pos_label=1)))
print("\n")
print(classification_report(test_y, pred_y))


# In[19]:


# Print out the Recall-Precision Plot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

plt.figure(figsize = (7,7))
plot_precision_recall_curve(logreg, test_x, test_y)
plt.title("Precision-Recall curve for Logistic Regression Classifier",fontsize=15)


# In[ ]:




