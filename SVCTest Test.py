import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import SVC
from numpy import genfromtxt


CD = genfromtxt('C:\\Users\Research\Research Data\SSCurveData.csv', delimiter=',')
CL = genfromtxt('C:\\Users\Research\Research Data\SSLabels.csv', delimiter=',')
TCD = genfromtxt('C:\\Users\Research\Research Data\SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('C:\\Users\Research\Research Data\SSLablesTest.csv', delimiter=',')
model = SVC()
model.fit(CD, CL)
score = model.score(TCD, TCL, sample_weight=None)
print(score)


