#dependencies
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import matplotlib.cm as cm
import numpy as np

# handle pickle IO
def read_pickle(pickle_name):
    path = "../pickles/"
    return pkl.load(open(path+pickle_name, "rb"))

def dump_pickle(obj, pickle_name):
    path = "../pickles/"
    pkl.dump(obj, open(path+pickle_name, "wb"))

def main():
	df = read_pickle("cleaned_data.pkl")

	words = []
	for doc in df['texts']:
	    for w in doc:
	        words.append(w)
	distinct_words = list(set(words))

	print (len(words))
	print (len(distinct_words))

	freq_dict = dict( zip(distinct_words, [0]*len(words)) )

	print("running cleaned_eda")
