##Importing Lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import random

#dataset
data=pd.read_csv("D:/DS/Data.csv",encoding='utf-8')
x=data.iloc[:,0:-1].values
y=data.iloc[:, 3].values


###taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])

##Encoding categorical data Independent variable
##Dummy Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x= LabelEncoder()
x[:,0] = labelEncoder_x.fit_transform(x[:,0])

onehotencoder = OneHotEncoder(categorical_features= [0])
x = onehotencoder.fit_transform(x).toarray()

##Encoding Dependent variable
## In Genral we will not do onehotencoder on Dependent variables

labelEncoder_y= LabelEncoder()
y= labelEncoder_y.fit_transform(y)


#### Splitting the data set into Training set and Test Set
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 0)

####Scaling##

from sklearn.preprocessing import StandardScaler 
sc_x= StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#### Final data preprocessing###

