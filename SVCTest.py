#Import
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt


#importing files to arrays
#Pre seperated training and test data
#CD = genfromtxt('C:\\Users\Research\Research Data\SSCurveData.csv', delimiter=',')
#CL = genfromtxt('C:\\Users\Research\Research Data\SSLabels.csv', delimiter=',')
#TCD = genfromtxt('C:\\Users\Research\Research Data\SSCurveDataTest.csv', delimiter=',')
#TCL = genfromtxt('C:\\Users\Research\Research Data\SSLablesTest.csv', delimiter=',')
#7k examples, not seperated
#X_data = genfromtxt('C:\\Users\Research\Research Data\sevenkData.csv', delimiter=',')
#Y_data = genfromtxt('C:\\Users\Research\Research Data\sevenkLabels.csv', delimiter =',')
#22k examples, not seperated
#X_data = genfromtxt('C:\\Users\Research\Research Data\entykData.csv', delimiter=',')
#Y_data = genfromtxt('C:\\Users\Research\Research Data\entykLabels.csv', delimiter =',')
#Full dataset
X_data = genfromtxt('C:\\Users\Research\Research Data\completeData.csv', delimiter=',')
Y_data = genfromtxt('C:\\Users\Research\Research Data\completeLabels.csv', delimiter =',')
#scaling and splitting data
#CD_scaled = preprocessing.scale(CD)
#TCD_scaled = preprocessing.scale(TCD)

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)

#Creating and training model
model = SVC(C= 1.0, kernel='poly', degree = 2)
model.fit(X_train, Y_train)



#scoring model
def score():
    predicted=model.predict(X_test)
    score = model.score(X_test, Y_test)
    print("normal score")
    print(score)

    print("accuracy_score")
    accscore=accuracy_score(Y_test, predicted)
    print(accscore)

    print("f1 score")
    f1=f1_score(Y_test, predicted, average='binary')
    print(f1)

    aps=average_precision_score(Y_test, predicted)
    print("average_precision_score")
    print(aps)

    print("hyperparameters")
    print(model.get_params())

score()