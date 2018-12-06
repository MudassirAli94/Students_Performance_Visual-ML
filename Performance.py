
# coding: utf-8

# In[1]:


# General Packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time

# General Mathematics package
import math as math

# Graphing Packages
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

# Statistics Packages
from scipy.stats import randint
from scipy.stats import skew

# Machine Learning Packages
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn import preprocessing
from skimage.transform import resize
import xgboost as xgb

# Neural Network Packages
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

# H2o packages
 import h2o
 from h2o.automl import H2OAutoML


# In[2]:


student = pd.read_csv("StudentsPerformance.csv")
student.head()

# We can create new columns in the dataframe to better our model and our visualization of the data
# In[3]:


student["overall score"] = np.nan
student["math score letter"] = np.nan
student["reading score letter"] = np.nan
student["writing score letter"] = np.nan
student["overall score letter"] = np.nan


# In[4]:


student["overall score"] = round((student["math score"] + 
                                  student["reading score"] + 
                                  student["writing score"])/3 , 2)


# In[5]:


student["math score letter"][student["math score"] <= 60] = "F"
student["math score letter"][np.logical_and
                             (student["math score"] > 60 , 
                              student["math score"] <= 69)] = "D"
student["math score letter"][np.logical_and
                             (student["math score"] >= 70 , 
                              student["math score"] <= 73)] = "C-"
student["math score letter"][np.logical_and
                             (student["math score"] >= 74 , 
                              student["math score"] <= 76)] = "C"
student["math score letter"][np.logical_and
                             (student["math score"] >= 77 , 
                              student["math score"] <= 79)] = "C+"
student["math score letter"][np.logical_and
                             (student["math score"] >= 80 , 
                              student["math score"] <= 83)] = "B-"
student["math score letter"][np.logical_and
                             (student["math score"] >= 84 , 
                              student["math score"] <= 86)] = "B"
student["math score letter"][np.logical_and
                             (student["math score"] >= 87 , 
                              student["math score"] <= 89)] = "B+"
student["math score letter"][np.logical_and
                             (student["math score"] >= 90 , 
                              student["math score"] <= 93)] = "A-"
student["math score letter"][np.logical_and
                             (student["math score"] >= 94 , 
                              student["math score"] <= 96)] = "A"
student["math score letter"][student["math score"] >= 97] = "A+"


# In[6]:


student["reading score letter"][student["reading score"] <= 60] = "F"
student["reading score letter"][np.logical_and
                             (student["reading score"] > 60 , 
                              student["reading score"] <= 69)] = "D"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 70 , 
                              student["reading score"] <= 73)] = "C-"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 74 , 
                              student["reading score"] <= 76)] = "C"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 77 , 
                              student["reading score"] <= 79)] = "C+"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 80 , 
                              student["reading score"] <= 83)] = "B-"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 84 , 
                              student["reading score"] <= 86)] = "B"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 87 , 
                              student["reading score"] <= 89)] = "B+"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 90 , 
                              student["reading score"] <= 93)] = "A-"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 94 , 
                              student["reading score"] <= 96)] = "A"
student["reading score letter"][student["reading score"] >= 97] = "A+"


# In[7]:


student["writing score letter"][student["writing score"] <= 60] = "F"
student["writing score letter"][np.logical_and
                             (student["writing score"] > 60 , 
                              student["writing score"] <= 69)] = "D"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 70 , 
                              student["writing score"] <= 73)] = "C-"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 74 , 
                              student["writing score"] <= 76)] = "C"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 77 , 
                              student["writing score"] <= 79)] = "C+"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 80 , 
                              student["writing score"] <= 83)] = "B-"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 84 , 
                              student["writing score"] <= 86)] = "B"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 87 , 
                              student["writing score"] <= 89)] = "B+"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 90 , 
                              student["writing score"] <= 93)] = "A-"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 94 , 
                              student["writing score"] <= 96)] = "A"
student["writing score letter"][student["writing score"] >= 97] = "A+"


# In[8]:


student["overall score letter"][student["overall score"] <= 60] = "F"
student["overall score letter"][np.logical_and
                             (student["overall score"] > 60 , 
                              student["overall score"] <= 69.99)] = "D"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 70 , 
                              student["overall score"] <= 73.99)] = "C-"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 74 , 
                              student["overall score"] <= 76.99)] = "C"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 77 , 
                              student["overall score"] <= 79.99)] = "C+"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 80 , 
                              student["overall score"] <= 83.99)] = "B-"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 84 , 
                              student["overall score"] <= 86.99)] = "B"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 87 , 
                              student["overall score"] <= 89.99)] = "B+"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 90 , 
                              student["overall score"] <= 93.99)] = "A-"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 94 , 
                              student["overall score"] <= 96.99)] = "A"
student["overall score letter"][student["overall score"] >= 97] = "A+"


# In[9]:


student.head()

# Now that we have added some necessary columns, we can begin doing data visualization
# In[10]:


for n in range(0,5):
    student.iloc[:,n] = pd.Categorical(student.iloc[:,n])
# 0 to 5 changes gender to test prepartion course into category
for n in range(-1,-5):
    student.iloc[:,n] = pd.Categorical(student.iloc[:,n])
# -1 to -5 changes math score letter to overall score letter to category
student['overall score'] = pd.to_numeric(student["overall score"])


# In[11]:


student.info()


# In[12]:


plt.figure(figsize= (20,5))
ax = plt.subplot(131)
pd.crosstab(student["math score letter"] , student.gender).plot(kind = 'bar', ax = ax)
plt.xticks(rotation = 360)
plt.title("Math Score" , size = 20)
plt.legend(loc=2, prop={'size': 15})

ax1 = plt.subplot(132)
pd.crosstab(student["reading score letter"] , student.gender).plot(kind = 'bar', ax = ax1)
plt.xticks(rotation = 360)
plt.title("Reading Score" , size = 20)
plt.legend(loc=2, prop={'size': 15})

ax2 = plt.subplot(133)
pd.crosstab(student["writing score letter"] , student.gender).plot(kind = 'bar', ax = ax2)
plt.title("Writing Score" , size = 20)
plt.xticks(rotation = 360)
plt.legend(loc=2, prop={'size': 15})

pd.crosstab(student["overall score letter"] , student.gender).plot(kind = 'bar' , figsize = (20,5))
plt.xlabel("overall score", size = 30)
plt.xticks(rotation = 360 , size  = 20)
plt.ylabel("count" , size = 20)
plt.legend(loc=2, prop={'size': 15})
plt.show()

# *From our graphs we can see that boys recieved a failing grade more than girls
# *Girls recieved a failing grade in math more than boys
# In[13]:


pd.crosstab(student["overall score letter"],
            student["parental level of education"]).plot.bar(figsize = (20,10))

plt.xlabel("overall score", size = 30)
plt.xticks(rotation = 360 , size  = 20)
plt.ylabel("count" , size = 30)
plt.title("Parents Degree and Grades" , size = 20)
plt.legend(loc=2, prop={'size': 15})
plt.show()


plt.figure(figsize = (20,8))
sns.boxplot(x="parental level of education", y="overall score", hue="gender" , 
            data=student )
plt.xticks(rotation=360 , size = 17)
plt.yticks( size = 20)
plt.xlabel("Level of Education of Parents", size = 20)
plt.ylabel("overall score" , size = 20)
plt.title("Parents Degree + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()

# *The children with the most F's came from the families where the parents highest level education completed was high school.

# *Those children who's parents had the highest education (master's degree) got the best overall scores
# In[14]:


pd.crosstab(student["overall score letter"],
            student["race/ethnicity"]).plot.bar(figsize = (30,15))

plt.xticks(rotation=360 , size = 30)
plt.yticks( size = 30)
plt.xlabel("overall score", size = 30)
plt.ylabel("count" , size = 30)
plt.title("Ethnicity/Race and Grades" , size = 30)
plt.legend(loc=2, prop={'size': 30})

plt.show()

plt.figure(figsize = (20,8))
sns.boxplot(x="race/ethnicity", y="overall score", hue="gender" , 
            data=student)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Ethnicity/Race", size = 20)
plt.ylabel("overall score" , size = 20)
plt.title("Ethnicity/Race + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()

# *The most F's came from Group C of the ethnicity/race
# *The best overall grades came from group E of the data.
# In[15]:


pd.crosstab(student["overall score letter"],
            student["lunch"]).plot.bar(figsize = (20,5))
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("overall score", size = 20)
plt.ylabel("count" , size = 20)
plt.title("Lunch and Grades" , size = 20)
plt.legend(loc=2, prop={'size': 20})
plt.show()

plt.figure(figsize = (20,8))
sns.boxplot(x="lunch", y="overall score", hue="gender" , 
            data=student)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Lunch", size = 20)
plt.ylabel("overall score" , size = 20)
plt.title("Lunch + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()

# *Those who receieved free/reduced lunch had the most F's
# *Those who received standard lunch did better on their exams than the ones with free/reduced lunch
# In[16]:


pd.crosstab(student["overall score letter"],
            student["test preparation course"]).plot.bar(figsize = (20,5))
plt.xticks(rotation = 360)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("overall score", size = 20)
plt.ylabel("count" , size = 20)
plt.title("Test Prep and Grades" , size = 20)
plt.legend(loc=2, prop={'size': 20})
plt.show()

plt.figure(figsize = (20,8))
sns.boxplot(x="test preparation course", y="overall score", hue="gender" , 
            data=student)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Test Prep", size = 20)
plt.ylabel("Overall Score" , size = 20)
plt.title("Test Preparation + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()

# *Those who took the test prep did better than the ones who did not
# In[17]:


sns.heatmap(student.corr() , square = True , annot = True , 
            linewidths=.8 , cmap="YlGnBu")
plt.title("Correlation Map")
plt.show()

# Based on the heat map there is a strong correlation between reading and writing score as opposed to math scores.
# In[18]:


sns.lmplot(x="reading score", y="writing score", hue="gender",
           data=student)
plt.show()

# Now we can perform our machine learning for the following questions:
# 1) If I did not make an overall score , can I correctly predict their overall score solely based on reading, writing and math scores? (I can predict the letter grade so we can make this into a classification problem)
# In[19]:


student_1 = student


# In[20]:


le = preprocessing.LabelEncoder()
for n in range(0,5):
    student_1.iloc[:,n] = le.fit_transform(student_1.iloc[:,n])


# In[21]:


for n in range(0,5):
    student_1.iloc[:,n] = pd.Categorical(student_1.iloc[:,n])
student_1['overall score'] = pd.to_numeric(student_1["overall score"])


# In[22]:


student_1["math score letter"] = pd.Categorical(student_1["math score letter"])
student_1["reading score letter"] = pd.Categorical(student_1["reading score letter"])
student_1["writing score letter"] = pd.Categorical(student_1["writing score letter"])
student_1["overall score letter"] = pd.Categorical(student_1["overall score letter"])


# In[23]:


student_2 = student_1


# In[24]:


student_1 = student_1.replace(["F" , "D" , "C-" , "C" , "C+", "B-" , "B" , "B+", "A-" , "A" , "A+"] , [0,1,2,3,4,5,6,7,8,9,10])

#    F = 0
#    D = 1
#    C- = 2
#    C = 3
#    C+ = 4
#    B- = 5
#    B = 6
#    B+ = 7
#    A- = 8
#    A = 9
#    A+ = 10


# In[25]:


student_1.head()


# In[26]:


X_score = student_1.drop(["overall score letter" , "overall score"] , axis = 1)
y_score = student_1["overall score letter"]

X_train, X_val, y_train, y_val = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)

# Logistic Regression: Predicting Overall Score
# In[27]:


logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_val)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print("accuracy",accuracy_score(y_val , y_pred)*100,"%")

# KNN: Predicting Overall Score
# In[28]:


param_grid = {'n_neighbors' : np.arange(1,50)}

knn_cv = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_cv, param_grid, cv = 5)

knn_cv.fit(X_train , y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred_knn = knn_cv.predict(X_val)

print("accuracy:" , accuracy_score(y_val , y_pred_knn)*100 , "%")

# Decision Tree: Predicting Overall Score
# In[29]:


tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred_DT = tree.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_DT))
print("accuracy:",round(accuracy_score(y_val , y_pred_DT)*100,2) , "%")

# Decision Tree + GridSearchCV: Predicting Overall Score
# In[30]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")

# Decision Tree + RandomizedSearchCV: Predicting overall score
# In[31]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")

# SVC: Predicting Overall Score
# In[32]:


warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_val)

print(classification_report(y_val , y_pred_SVC))
print('accuracy:' , accuracy_score(y_val , y_pred_SVC)*100,"%")

# XGBoost: Predicting Overall Score
# In[33]:


student_int = student_1[0:-1].astype("int64")

X_score_int = student_int.drop(["overall score letter" ,"overall score"] , axis = 1)
y_score_int = student_int["overall score letter"]

X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(X_score_int, y_score_int, test_size = 0.3, random_state=42)


# In[34]:


warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_val_int)

print("accuracy:",round(accuracy_score(y_val_int, y_pred_XGB)*100,2) , "%")

# The accuracy I obtained is fairly good when trying to predict their overall scores. Our best accuracy is 86% via randomizedsearch cv with deceision tree.

# 2) Can I get a better accuracy if I reduced the number of categories?
# In[35]:


student_2.replace(["C-" , "C" , "C+" ,] , ("C") , inplace = True)
student_2.replace(["B-" , "B" , "B+" ,] , ("B") , inplace = True)
student_2.replace(["A-" , "A" , "A+" ,] , ("A") , inplace = True)


# In[36]:


student_2.replace(["F" , "D" , "C" , "B" , "A"] , (0,1,2,3,4) , inplace = True)

#    F = 0
#    D = 1
#    C = 2
#    B = 3
#    A = 4


# In[37]:


X_score = student_2.drop(["overall score letter" , "overall score"] , axis = 1)
y_score = student_2["overall score letter"]

X_train, X_test, y_train, y_test = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)

# Logistic Regression: Predicting overall score letter with reduced categories
# In[38]:


logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred))
print("accuracy",accuracy_score(y_test , y_pred)*100,"%")

# Decision Tree: Predicting overall score letter with reduced categories
# In[39]:


tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred_DT = tree.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_DT))
print("accuracy:",round(accuracy_score(y_test , y_pred_DT)*100,2) , "%")

# Decision Tree + GridsearchCV: Predicting overall score letter with reduced categories
# In[40]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_RF))
print("accuracy:",round(accuracy_score(y_test , y_pred_RF)*100,2),"%")

# Decision Tree + RandomizedSearchCV: Predicting overall score letter with reduced categories
# In[41]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_RF))
print("accuracy:",round(accuracy_score(y_test , y_pred_RF)*100,2),"%")

# SVC: Predicting overall score letter with reduced categories
# In[42]:


warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_test)

print(classification_report(y_test , y_pred_SVC))
print('accuracy:' , round(accuracy_score(y_test , y_pred_SVC)*100,2),"%")

# XGBoost: Predicting overall score letter with reduced categories
# In[43]:


student_int = student_2[0:-1].astype("int64")

X_score_int = student_int.drop(["overall score letter" ,"overall score"] , axis = 1)
y_score_int = student_int["overall score letter"]

X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_score_int, y_score_int, test_size = 0.3, random_state=42)


# In[44]:


warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_test_int)

print("accuracy:",round(accuracy_score(y_test_int, y_pred_XGB)*100,2) , "%")

# Reducing the number of categories significantly increased our accuracy, our best accuracy being 91% via XGBoost

# 4) What will be our R^2 value if we performed regression instead of classification?
# In[45]:


X_score = student_1.drop(["overall score letter" , "overall score"] , axis = 1)
y_score = student_1["overall score"]

X_train, X_test, y_train, y_test = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)

# Linear Regression: Predicting overall numerical score
# In[46]:


reg = linear_model.LinearRegression()
reg.fit(X_train , y_train)
y_pred_reg = reg.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test,y_pred_reg))
print("RMSE: " , RMSE)
MSE = mean_squared_error(y_test,y_pred_reg )
print("MSE: " ,MSE)
print("R^2: " , r2_score(y_test, y_pred_reg))

# Ridge Regression: Predicting overall numerical score
# In[47]:


ridge = Ridge(alpha = 0.4 , normalize = True)

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("RMSE: " , np.sqrt(mean_squared_error(y_test,ridge_pred)))
print("MSE: " , mean_squared_error(y_test,ridge_pred))
print("R^2: " , r2_score(y_test, ridge_pred))

# Lasso Regression: Predicting overall numerical score
# In[48]:


lasso = Lasso(alpha = 0.4, normalize = True , max_iter = 1000000)

lasso.fit(X_train,y_train)

lasso_pred = ridge.predict(X_test)
print("RMSE: " , np.sqrt(mean_squared_error(y_test,lasso_pred)))
print("MSE: " , mean_squared_error(y_test,lasso_pred))
print("R^2: " , r2_score(y_test, lasso_pred))

# Regression gives us a high R^2 value which is also a noteable method for our machine learning practices.4) As I saw in the graph earlier, girls did better on their score than boys, can we obtain a good accuracy on predicting whether the student is a girl or boy?
# In[49]:


X_score = student_1.drop(["gender"] , axis = 1)
y_score = student_1["gender"]

X_train, X_val, y_train, y_val = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)

# Logistic Regression: Predicting Gender
# In[50]:


logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_val)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print("accuracy",round(accuracy_score(y_val , y_pred)*100,2),"%")

# KNN: Predicting Gender
# In[51]:


param_grid = {'n_neighbors' : np.arange(1,50)}

knn_cv = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_cv, param_grid, cv = 5)

knn_cv.fit(X_train , y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred_knn = knn_cv.predict(X_val)

print("accuracy:" , round(accuracy_score(y_val , y_pred_knn)*100,2) , "%")

# Decreasing the number of categories increased our accuracy.Decision Tree: Predicting Gender
# In[52]:


tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred_DT = tree.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_DT))
print("accuracy:",accuracy_score(y_val , y_pred_DT)*100 , "%")

# Decision Tree + GridSearch: Predicting Gender
# In[53]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 5 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")

# Decision Tree + RandomSearch: Predicting Gender
# In[54]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 5 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")

# SVC: Predicting Gender
# In[55]:


warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_val)

print(classification_report(y_val , y_pred_SVC))
print('accuracy:' , round(accuracy_score(y_val , y_pred_SVC)*100,2),"%")

# XGBoost: Predicting Gender
# In[56]:


student_int = student_1[0:-1].astype("int64")

X_score_int = student_int.drop(["gender"] , axis = 1)
y_score_int = student_int["gender"]

X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(X_score_int, y_score_int, test_size = 0.3, random_state=42)


# In[57]:


warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_val_int)

print("accuracy:",accuracy_score(y_val_int, y_pred_XGB)*100 , "%")

# Overall we were able to get 90% accuracy in predicting the gender via Logistic Regression.4) 
# Can we use unsupervised learning (such as KMeans), to predicting gender based on reading and math scores? 
# Unsupervised Learning: KMeans Clustering
# In[58]:


student = pd.read_csv("StudentsPerformance.csv")


# In[59]:


from sklearn.cluster import KMeans

gender = student_1['gender'].values
X_gen = student_1.drop('gender' , axis = 1).values


# In[60]:


model = KMeans(n_clusters = 2)
model.fit(X_gen)
labels = model.predict(X_gen)


# In[61]:


xs = X_gen[:,4]
ys = X_gen[:,5]

f, (ax1, ax2) = plt.subplots(figsize=(20, 10) , ncols = 2)

sns.set()

sns.scatterplot(x="math score", y="reading score", hue="gender" , data=student , ax = ax1)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Math Score", size = 20)
plt.ylabel("Reading Score" , size = 20)
plt.title("Actual Gender Data" , size = 20)
#plt.legend(loc=2, prop={'size': 20})

sns.scatterplot(x = xs,y = ys , hue = labels , ax=ax2)
plt.xlabel("Math Score" , size = 20)
plt.ylabel("Reading Score" , size = 20)
plt.title("KMeans Predicting Gender" , size = 20)
plt.legend(loc=2, prop={'size': 20})
plt.show()

acc = round(accuracy_score(labels , student_1.gender)*100,2)
print(('accuracy: '+ str(acc)+" %").center(125))

# Overall we can use unsupervised learning to predict gender. However, KMeans clustering only gave us a poor accuracy, the best method for such classification would be using supervised learning.