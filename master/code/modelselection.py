#dependencies
import glob
import pandas as pd
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
import operator
from sklearn.model_selection import StratifiedKFold



print("selecting model...") 

# load pickle
# pickle must be of type DataFrame
def read_pickle(pickle_name):
    path = "../pickles/"
    obj = pkl.load(open(path+pickle_name, "rb"))
    if not type(obj) == type(pd.DataFrame()):
        raise TypeError("object to read must be DataFrame")
    return obj

# dumps a pickle of obj
# only DataFrames permitted
def dump_pickle(obj, pickle_name):
    path = "../pickles/"
    if not type(obj) == type(pd.DataFrame()):
        raise TypeError("object to dump must be DataFrame")
    pkl.dump(obj, open(path+pickle_name, "wb"))



#classifiers = ['svm','knn','rf','lr']
#kernel = ['rbf','linear','poly','sigmoid']
#solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#data = read_pickle("pickle name")
def k_folds_cross_validation(data, kfolds, classifiers, kernel, solvers):
    sv = defaultdict(list)
    knn = defaultdict(list)
    rf = defaultdict(list)
    lr = defaultdict(list)

    #print(data)

    #train and test data generation
    X = np.array(data.drop(['labels'],1))
    y = np.array(list(map(int,data['labels'])))

    #split for cross validation
    split = int(data.shape[0]/kfolds)
    
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True)

    for (train_indices,val_indices) in skf.split(X,y):

        #index values for the testing set
        x_train, x_test = X[train_indices], X[val_indices]
        y_train, y_test = y[train_indices], y[val_indices]
        
        for classifier in classifiers:
            if classifier == 'svm':
                for ker in kernel:
                    sv_classifier = svm.SVC(kernel=ker, class_weight = 'balanced')
                    sv_classifier.fit(x_train, y_train)
                    temp = sv_classifier.score(x_test,y_test)
                    sv[classifier+"_"+ker].append(temp)

                   
            #play with upper range
            if classifier == 'knn':
                for n in range(5,50,5):
                    knc = KNeighborsClassifier(n_neighbors=n)
                    knc.fit(x_train, y_train)
                    temp = knc.score(x_test,y_test)
                    knn[classifier+"_"+str(n)].append(temp)
                    
            #play with upper range   
            if classifier == 'rf':
                for n in range(5,50,5):
                    rfc = RandomForestClassifier(n_estimators=n, class_weight = 'balanced')
                    rfc.fit(x_train, y_train)
                    temp = rfc.score(x_test,y_test)
                    rf[classifier+"_"+str(n)].append(temp)


            if classifier == 'lr':
                for sol in solvers:
                    log_reg = LogisticRegression(solver = sol, class_weight = 'balanced')
                    log_reg.fit(x_train, y_train)
                    temp = log_reg.score(x_test,y_test)
                    lr[classifier+"_"+sol].append(temp)

    
    scores = [dict(sv),dict(knn),dict(rf),dict(lr)]
    
    average_scores = {}
    for model in scores:
        for params in model:
            average_scores[params] = sum(model[params])/float(10)

    return average_scores

def model_selection(average_scores):
    sorted_dict = sorted(average_scores.items(), key=operator.itemgetter(1), reverse=True)
    selected_model = str(sorted_dict[0][0])
    top_average_accuracy = str(sorted_dict[0][1]*100) + "%"
    print("selected model is " + selected_model +
          " with average accuracy: " + top_average_accuracy)
    return selected_model.split("_")

def main():    
    data = read_pickle("doc_vecs.pkl")

    ##print(data.head())
    model,hyperparameter = model_selection(k_folds_cross_validation(data, 10,
                                     ['svm','knn','rf','lr'],
                                     ['rbf','linear','poly','sigmoid'],
                                     ['newton-cg', 'lbfgs', 'liblinear']))

    print("selected model is ", model)
    print("hyperparameter for model is ", hyperparameter)
    pkl.dump(model,open("../pickles/best_model.pkl","wb"))
    pkl.dump(hyperparameter,open("../pickles/hyperparameter.pkl","wb"))
        
    #print(data.shape[0])
    #kfolds = 10
    #print(result)


    #Major changes required:


    
