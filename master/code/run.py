import cleaning
import cleaned_eda
import train_test_split
import preprocessing
import preprocessed_eda
import modelselection
import training
import testing
import results

import pandas as pd

#4,5,6,7,8 categories
#no stopwords, standard stopwords, custom stopwords
#556, 200
#class weight balanced, null
cat_list =['4CAT','5CAT','6CAT','7CAT','8CAT']
n_size_list=[556,200]
stopwords_options = [0,1,2]
cols = ['model','categories', 'sample size', 'accuracy','confustion matrix','accuracy report','stopwords', 'class weight']
def looper():
	file_results =[]
	for cat in cat_list:
		for n_size in n_size_list:
			for stpwords in stopwords_options:
				print("in it")
				cleaning.main(cat,n_size,stpwords)
				cleaned_eda.main()
				train_test_split.main()
				preprocessing.main()
				#preprocessed_eda.main()
				modelselection.main()
				training.main()
				accuracy = testing.main()
				results.main(cat,n_size,stpwords,'balanced',accuracy,file_results)
	df = pd.DataFrame(file_results,columns=cols)
	df.to_csv('balanced_new_results.csv')

looper()
    
