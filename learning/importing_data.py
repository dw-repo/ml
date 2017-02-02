# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:51:32 2016

@author: Darbinyan
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd
from io import StringIO

#
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3,
    random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#------------------------------------

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df.columns = ['color','size','price','classlabel']

#-----------------------------------

stdsc=StandardScaler()
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
df_wine.columns = ['Class label', 'Alcohol',
'Malic acid','Ash',
'Alcalinity of ash','Magnesium',
'Total phenols', 'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity','Hue',
'OD280/OD315 of diluted wines',
'Proline']

print ('Class labels', np.unique(df_wine['Class label']))

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)