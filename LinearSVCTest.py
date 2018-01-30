import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from numpy import genfromtxt

#importing files to arrays
CD = genfromtxt('SSCurveData.csv', delimiter=',')
CL = genfromtxt('SSLabels.csv', delimiter=',')
TCD = genfromtxt('SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('SSLablesTest.csv', delimiter=',')

#scaling data
CD_scaled = preprocessing.scale(CD)
TCD_scaled = preprocessing.scale(TCD)

#Test line for commit
#Creating and training model
model = LinearSVC()
model.fit(CD_scaled, CL)

#scoring model
score = model.score(TCD_scaled, TCL)
print(score)


