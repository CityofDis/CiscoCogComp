#confussion matrix and classification report
#test_doc_vecs
#dependencies
import glob
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

def main(cat,n_size,stpwords,class_weight,accuracy,file_results):
	model = pkl.load(open("../pickles/trained_model.pkl","rb"))
	data = read_pickle("test_doc_vecs.pkl")

	temp = read_pickle("cleaned_test_data.pkl")

	data['labels'] = list(temp['labels'])

	#confusion matrix
	temp = np.array(data.drop(['labels'],1))
	y_pred = model.predict(temp)
	conf_mat = confusion_matrix(data['labels'] ,y_pred)
	print("Confusion Matrix:\n%s\n"%conf_mat)

	report = classification_report(data['labels'], y_pred)
	print("Classification Report:\n%s\n"%report)

	file_results.append([model,cat,n_size,accuracy,conf_mat,report,stpwords,class_weight])
	#cols = ['model','categories', 'sample size', 'accuracy','confustion matrix','accuracy report','stopwords', 'class weight']
	

