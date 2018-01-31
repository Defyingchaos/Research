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
CD = genfromtxt('C:\\Users\Research\Research Data\SSCurveData.csv', delimiter=',')
CL = genfromtxt('C:\\Users\Research\Research Data\SSLabels.csv', delimiter=',')
TCD = genfromtxt('C:\\Users\Research\Research Data\SSCurveDataTest.csv', delimiter=',')
TCL = genfromtxt('C:\\Users\Research\Research Data\SSLablesTest.csv', delimiter=',')
print("data imported")

#scaling data
CD_scaled = preprocessing.scale(CD)
TCD_scaled = preprocessing.scale(TCD)
print("data scaled")

#Parameters and distributions
model=SVC(kernel='poly')
param_dist = {'C': sp_randint(2, 4), 'degree': sp_randint(1, 2)}
print("model created")

#Creating and training model
n_iter_search = 20
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search)
#model = SVC(C= 1.0, kernel='poly', degree = 2)
#model.fit(CD, CL)
random_search.fit(CD, CL)
print("model fit")

#scoring model
score = random_search.score(TCD, TCL)
print(score)
print(model.get_params())