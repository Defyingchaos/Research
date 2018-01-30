import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import SVC
from numpy import genfromtxt


CD = genfromtxt('SSCurveData.csv', delimiter=',')
CL = genfromtxt('SSLabels.csv', delimiter=',')
TCD = genfromtxt('SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('SSLablesTest.csv', delimiter=',')
model = SVC()
model.fit(CD, CL)
score = model.score(TCD, TCL, sample_weight=None)
print(score)


