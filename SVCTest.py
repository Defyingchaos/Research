#Import
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt

#importing files to arrays
CD = genfromtxt('SSCurveData.csv', delimiter=',')
CL = genfromtxt('SSLabels.csv', delimiter=',')
TCD = genfromtxt('SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('SSLablesTest.csv', delimiter=',')

#scaling data
CD_scaled = preprocessing.scale(CD)
TCD_scaled = preprocessing.scale(TCD)

#Creating and training model
model = SVC(C= 1.0, kernel='poly', degree = 2)
model.fit(CD, CL)

#scoring model
score = model.score(TCD, TCL)
print("normal score")
print(score)
print("accuracy_score")
predicted=model.predict(TCD)
jss=accuracy_score(TCL, predicted)
print(jss)
print("f1 score")
f1=f1_score(TCL, predicted, average='binary')
print(f1)
print("hyperparameters")
print(model.get_params())