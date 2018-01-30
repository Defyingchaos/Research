#Import a lot
print("test")
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import randint as sp_randint

#importing files to arrays
CD = genfromtxt('SSCurveData.csv', delimiter=',')
CL = genfromtxt('SSLabels.csv', delimiter=',')
TCD = genfromtxt('SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('SSLablesTest.csv', delimiter=',')
print("data imported")

#scaling data
CD_scaled = preprocessing.scale(CD)
TCD_scaled = preprocessing.scale(TCD)
print("data scaled")

#Parameters and distributions
model=SVC(kernel='poly')
param_grid = {'C': [1,10], 'degree': [0, 5], 'gamma': [0,2], 'coef0': [0,5]}

print("model created")

#Creating and training model
grid_search = GridSearchCV(model, param_grid=param_grid)
#random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search)
#model = SVC(C= 1.0, kernel='poly', degree = 2)
#model.fit(CD, CL)
grid_search.fit(CD, CL)
print("model fit")

#scoring model
score = grid_search.score(TCD, TCL)
print(score)
print(model.get_params())