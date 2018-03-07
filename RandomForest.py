#Import
import numpy as np
import sklearn
import pandas as pd
from scipy.stats import randint, expon 
#from sklearn.svm import SVC
#from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt

import json
import gspread
import oauth2client
from oauth2client.client import SignedJwtAssertionCredentials


#importing files to arrays
#1k Set 1Pre seperated training and test data
X_data = pd.read_csv("C:\\Users\Research\Research Data\SSCurveData.csv")
Y_data = pd.read_csv("C:\\Users\Research\Research Data\SSLabels.csv")
#1k Set 2
#X_data = pd.read_csv("C:\\Users\Research\Research Data\SSCurveDataTest.csv")
#Y_data = pd.read_csv("C:\\Users\Research\Research Data\SSLablesTest.csv")
#7k examples, not seperated
#X_data = pd.read_csv("C:\\Users\Research\Research Data\sevenkData.csv")
#Y_data = pd.read_csv("C:\\Users\Research\Research Data\sevenkLabels.csv")
#22k examples, not seperated
#X_data = pd.read_csv("C:\\Users\Research\Research Data\entykData.csv")
#Y_data = pd.read_csv("C:\\Users\Research\Research Data\entykLabels.csv")
#Full dataset
#X_data = pd.read_csv("C:\\Users\Research\Research Data\completeData.csv")
#Y_data = pd.read_csv("C:\\Users\Research\Research Data\completeLabels.csv")

#scaling and splitting data
#X_data = preprocessing.scale(X_data)
#Y_data = preprocessing.scale(Y_data)
print("done")

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)

#Exhaustive grid search parameters
param_grid = {}

#Randomized grid search parameters
param_dist = {}
n_estimators = 40


#Creating and training model
    #Regular model
#model = SVC(C= 10, gamma= .0001, kernel= 'rbf', verbose = True)
model = RFC(n_estimators, verbose = True)
    #Regular grid search
#model = GridSearchCV(SVC, param_grid=param_grid)
#searchtype = GridSearch
    #Randomized Grid search
    #model = RandomizedSearchCV(SVC, param_distributions=param_dist, n_iter=n_iter_search)
    #searchtype = "Randomized, n_iter_search = {}".format(n_iter_search)
    #Training
model.fit(X_train, Y_train.values.ravel())

    #Cross Validation
    #scores = cross_val_score(model, X_data, Y_data.values.ravel(), cv = 5)

#finds and prints model scores and statistics
def score():
    predicted=model.predict(X_test)
    #print(((((len(predicted)-(sum(predicted)))/2)+sum(predicted)))/len(predicted))
    score = model.score(X_test, Y_test)
    print("normal score")
    print(score)

    print("accuracy_score")
    accscore=accuracy_score(Y_test, predicted)
    print(accscore)

   # print("Cross validation score and 95% CI")
   # CVSmean = scores.mean()
   # print(CVSmean)
   # CI = scores.std()*2
   # print(CI)

    print("f1 score")
    f1=f1_score(Y_test, predicted, average='binary')
    print(f1)

    aps=average_precision_score(Y_test, predicted)
    print("average_precision_score")
    print(aps)

   # print("model hyperparameters")
   # params = model.best_params_
   # print(params)

    tn, fp, fn, tp = confusion_matrix(Y_test, predicted).ravel()
    print(tn, fp, fn, tp)

   # save_scores(accscore, CVSmean, CI, f1, aps, params, searchtype)
    save_scores(accscore, "NA", "NA", f1, aps, "NA", "NA")
#uploads scores to google drive
def save_scores(accscore, CVSmean, CI, f1, aps, params, searchtype):
    json_key = json.load(open('creds.json')) 
    scope = ['https://spreadsheets.google.com/feeds']

    credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

    file = gspread.authorize(credentials)
    sheet = file.open('Data').sheet1

    nar = next_availible_row(sheet)

    sheet.update_acell("A{}".format(nar), len(Y_data)+1)
    sheet.update_acell("B{}".format(nar), accscore)
    sheet.update_acell("C{}".format(nar), CVSmean)
    sheet.update_acell("D{}".format(nar), CI)
    sheet.update_acell("E{}".format(nar), f1)
    sheet.update_acell("F{}".format(nar), aps)
    sheet.update_acell("G{}".format(nar), params)
    sheet.update_acell("H{}".format(nar), searchtype)
    sheet.update_acell("I{}".format(nar), "SVC")
   #sheet.update_acell('A1', 'Dataset Size')
    #print(sheet.row_count)

#finds next open row in spreadsheet
def next_availible_row(sheet):
    str_list = list(filter(None, sheet.col_values(1)))
    return str(len(str_list)+1)


score()