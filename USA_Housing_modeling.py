#Importing the Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset:
usa_housing=pd.read_csv('C:/Users/mizoh/Desktop/Data Glacier/USA_Housing/USA_Housing.csv')
X=usa_housing.iloc[:,:-2]
y=usa_housing.iloc[:,5]

#Splitting the dataset into training set and test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#Linear Regression modeling:
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)

#Pickling:
import pickle
pickle.dump(regressor,open('USA_Housing.pkl','wb'))