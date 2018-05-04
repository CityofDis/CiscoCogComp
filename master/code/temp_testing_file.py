import pandas as pd
import pickle as pkl

temp = pkl.load(open("../pickles/cleaned_test_data.pkl","rb"))
print(temp['labels'])
