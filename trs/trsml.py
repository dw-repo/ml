# -*- coding: utf-8 -*-
"""
Fri Nov 25 13:45:26 2016
Testing TRS with machine learning

@author: Darbinyan 
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm


df = pd.read_csv('C:\\Users\\Darbinyan.FSPL\\Documents\\MATLAB\\TRS_ML\\cyclone_number_upto2014.csv',header = 0)
cols = ['soi4','soi5','soi6','soi7','soi8','wp4','wp5','wp6','wp7','wp8','solar3','solar4','solar5','solar6','solar7','solar8']
colscomp = ['soi5','soi6','soi7','wp5','wp6','wp7','solar5','solar6','solar7','IB_WMO']

X = df.get(cols).values
y = df.get(['IB_WMO']).values

#sns.set(style='whitegrid', context='notebook')
#sns.pairplot(df[colscomp], size=2.5)
#plt.show()

#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cm,
#                 cbar=True,
#                 annot=True,
#                 square=True,
#                 fmt='.2f',
#                 annot_kws={'size': 15},
#                 yticklabels=cols,
#                 xticklabels=cols)
#plt.show()                 

#X = df.get(['soi-2','soi-1','sst-2','sst-1','wp-2','wp-1','sol-2','sol-1']).values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=1)

 

## standard scaler
#sc=StandardScaler()
#sc.fit(X_train)
#X_train_std = sc.transform(X_train)
#X_test_std = sc.transform(X_test)

## random forest
#forest = RandomForestRegressor(
#                               n_estimators=1000,
#                               criterion='mse',
#                               random_state=1,
#                               n_jobs=-1)
#forest.fit(X_train,y_train)
#y_train_pred = forest.predict(X_train)
#y_test_pred = forest.predict(X_test)

##----------SVR start------------
### svr
#clf = svm.SVR(kernel = 'rbf', C=1, gamma = 0.01)
#clf.fit(X_train, y_train.squeeze())
#y_train_pred = clf.predict(X_train)
#y_test_pred = clf.predict(X_test)

## Check C and gamma
#Citer = np.linspace(0.1,10,100)
#maeCtest_train = []
#maeCtest_test = [] 
#for i in Citer:
#    
#    clf = svm.SVR(kernel = 'rbf', C=i, gamma = 10e6)
#    clf.fit(X_train, y_train.squeeze())
#    y_train_pred = clf.predict(X_train)
#    y_test_pred = clf.predict(X_test)
#    
#    maeCtest_train.append(mae(y_train, y_train_pred))
#    maeCtest_test.append(mae(y_test, y_test_pred))
#plt.figure()
#plt.plot(Citer, maeCtest_train)    
#plt.plot(Citer, maeCtest_test) 
#
#
#
#gamiter = np.linspace(0.0001,0.1,100)
#maeGamtest_train = []
#maeGamtest_test = [] 
#for i in gamiter:
#    
#    clf = svm.SVR(kernel = 'rbf', C=1, gamma = i)
#    clf.fit(X_train, y_train.squeeze())
#    y_train_pred = clf.predict(X_train)
#    y_test_pred = clf.predict(X_test)
#    
#    maeGamtest_train.append(mae(y_train, y_train_pred))
#    maeGamtest_test.append(mae(y_test, y_test_pred))
#plt.figure()
#plt.plot(gamiter, maeGamtest_train)    
#plt.plot(gamiter, maeGamtest_test) 

## Check C and gamma together  for SVR with rbf
#-----
Citer = np.linspace(0.00001,20.001,40,endpoint=False)
gamiter = np.linspace(0.0001,0.1001,50,endpoint=False)

maetesting_train = []
maetesting_test = [] 

foldnm = 'C:\\Users\\Darbinyan.FSPL\\Documents\\pypersonal_packages\\ml\\trs\\svr_rbf_test\\run3\\'
f=open(foldnm+'svr_test.csv', 'w')
f.write(np.array2string(gamiter,max_line_width=600,precision=5,separator=',')+'\n')
    
for i in Citer:
    maetesting_train = []
    maetesting_test = []
    f.write(np.array2string(i)+',')
    for j in gamiter:
        clf = svm.SVR(kernel = 'rbf', C=i, gamma = j)
        clf.fit(X_train, y_train.squeeze())
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        maetesting_train.append(mae(y_train, y_train_pred))
        maetesting_test.append(mae(y_test, y_test_pred))
        
        f.write(str(mae(y_test, y_test_pred))+',')
    
    f.write('\n')    
    
    plt.figure(figsize=(10,8))
    lin1=plt.plot(gamiter, maetesting_train)
    lin2=plt.plot(gamiter, maetesting_test)
    plt.xlabel('Gamma')
    plt.ylabel('MAE')
    plt.title('Testing SVR C='+np.array2string(i))
    plt.legend(['Trainig set','Test set'])
    plt.savefig(foldnm+'test_SVR_C_'+np.array2string(i,precision=2).replace('.','p'))
    plt.close()

f.close()    
#----

## Check C for SVR with poly kernels
#Citer = np.linspace(0.001,10.001,10,endpoint=True)
#deg = 2
#
#maetesting_train = []
#maetesting_test = [] 
#
#foldnm = 'C:\\Users\\Darbinyan.FSPL\\Documents\\pypersonal_packages\\ml\\trs\\svr_poly_test\\'
#f=open(foldnm+'svr_poly_test_'+str(deg)+'.csv', 'w')
##f.write(np.array2string(gamiter,max_line_width=400,precision=5,separator=',')+'\n')
#maetesting_train = []
#maetesting_test = []    
#for i in Citer:
#    print(i)
#    f.write(np.array2string(i)+',')
#    
#    clf = svm.SVR(kernel = 'poly', C=i, degree = deg)
#    clf.fit(X_train, y_train.squeeze())
#    y_train_pred = clf.predict(X_train)
#    y_test_pred = clf.predict(X_test)
#        
#    maetesting_train.append(mae(y_train, y_train_pred))
#    maetesting_test.append(mae(y_test, y_test_pred))
#        
#    f.write(str(mae(y_train, y_train_pred))+',')
#    f.write(str(mae(y_test, y_test_pred)))
#    f.write('\n')
#    
#plt.figure(figsize=(10,8))
#plt.plot(Citer, maetesting_train)
#plt.plot(Citer, maetesting_test)
#plt.xlabel('C')
#plt.ylabel('MAE')
#plt.title('Testing SVR with poly kernel degree='+str(deg))
#plt.legend('Trainig set','Test set')
#plt.savefig(foldnm+'test_SVR_poly_deg_'+str(deg))
#plt.close()
#
#f.close()    


##----------SVR end--------------    
#
## check agains average
#y_train_pred = y_train.mean().repeat(len(y_train))
#y_test_pred = y_test.mean().repeat(len(y_test))

print('MSE train: %.3f, test %.3f' % (
       mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test %.3f' % (
       r2_score(y_train, y_train_pred),
       r2_score(y_test, y_test_pred)))

print('MAE train: %.3f, test %.3f' % (
       mae(y_train, y_train_pred),
       mae(y_test, y_test_pred)))
