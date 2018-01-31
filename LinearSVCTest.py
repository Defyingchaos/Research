import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from numpy import genfromtxt
import os
import sys
sys.path.append('C:\\Users\Research\Research Data')


#importing files to arrays
CD = genfromtxt('C:\\Users\Research\Research Data\SSCurveData.csv', delimiter=',')
CL = genfromtxt('C:\\Users\Research\Research Data\SSLabels.csv', delimiter=',')
TCD = genfromtxt('C:\\Users\Research\Research Data\SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('C:\\Users\Research\Research Data\SSLablesTest.csv', delimiter=',')

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


