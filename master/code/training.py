#dependencies
import glob
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys



print("training on data...")



def read_pickle(pickle_name):
    path = "../pickles/"
    return pkl.load(open(path+pickle_name, "rb"))

def dump_pickle(obj, pickle_name):
    path = "../pickles/"
    pkl.dump(obj, open(path+pickle_name, "wb"))

def select_best_classifier_and_hyperparameter(classifier,hyperparameter):
    classifier_dict = {"svm":"svm.SVC","knn":"KNeighborsClassifier",
                       "rf":"RandomForestClassifier","lr":"LogisticRegression"}

    parameter_names = {"svm":"kernel","knn":"n_neighbours",
                       "rf":"n_estimators","lr":"solver"}

    object_conversions = {"svm":sklearn.svm.classes.SVC,
                          "knn":sklearn.neighbors.classification.KNeighborsClassifier ,
                          "lr": sklearn.linear_model.logistic.LogisticRegression ,
                          "rf":sklearn.ensemble.forest.RandomForestClassifier }
    
    if classifier == "knn":
      classifier_with_hyperparameter = classifier_dict[classifier] + "(" + parameter_names[classifier] + "=" + str(hyperparameter) + ")"
    if classifier == "rf":
      classifier_with_hyperparameter = classifier_dict[classifier] + "(" + parameter_names[classifier] + "=" + str(hyperparameter) +","+ "class_weight = 'balanced'"+")"
    if classifier == "lr":
      classifier_with_hyperparameter = classifier_dict[classifier] + "(" + parameter_names[classifier] + "=" + str(hyperparameter) +","+ "class_weight = 'balanced'"+")"
    if classifier == "svm":
      classifier_with_hyperparameter = classifier_dict[classifier] + "(" + parameter_names[classifier] + "=" + str(hyperparameter) +","+ "class_weight = 'balanced'"+")"




    return object_conversions[classifier](classifier_with_hyperparameter)

    
def train(train_data_features, train_data_labels, classifier,hyperparameter):
  
  if classifier == 'lr':
    model = LogisticRegression(solver = str(hyperparameter), class_weight = 'balanced')

  if classifier == 'rf':
    model = RandomForestClassifier(n_estimators = int(hyperparameter), class_weight = 'balanced')

  if classifier == 'knn':
    model = KNeighborsClassifier(n_neighbors = int(hyperparameter))

  if classifier == 'svm':
    model = svm(kernel = str(hyperparameter),class_weight = 'balanced')

  fit = model.fit(train_data_features,train_data_labels)
  return fit

def main():
    data = read_pickle("doc_vecs.pkl")
    X = np.array(data.drop(['labels'],1))
    y = np.array(list(map(int,data['labels'])))
    #classifier = select_best_classifier_and_hyperparameter(read_pickle("best_model.pkl"),
                                                           #read_pickle("hyperparameter.pkl"))
    model = train(X,y,read_pickle("best_model.pkl"),read_pickle("hyperparameter.pkl"))
    dump_pickle(model,"trained_model.pkl")
    print("finished training")


