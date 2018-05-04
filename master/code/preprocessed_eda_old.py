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

def plot(tsne_values, mapped_labels):
    index = range(len(tsne_values))
    columns = ['X', 'Y', 'labels']
    tsne_df = pd.DataFrame(columns = columns, index = index)
    tsne_df['X'] = [i[0] for i in tsne_values]
    tsne_df['Y'] = [i[1] for i in tsne_values]
    tsne_df['labels'] = mapped_labels
    
    #color map
    classes = list(set(mapped_labels))
    num_classes = len(classes)
    cvals = cm.rainbow(np.linspace(0, 1, num_classes))
    color_dict = dict(zip(classes, cvals))
    colors = [color_dict[l] for l in mapped_labels]
    plt.scatter(tsne_df['X'], tsne_df['Y'], c = colors)
    plt.show()

if __name__ == "__main__":
    tfidf_tsne = read_pickle("tfidf_tsne.pkl")
    d2v_tsne = read_pickle("d2v_tsne.pkl")
    weighted_d2v_tsne = read_pickle("weighted_d2v_tsne.pkl")
    labels = read_pickle("cleaned_train_data.pkl")['labels']
    plot(d2v_tsne, labels)
    plot(weighted_d2v_tsne,labels)
    plot(tfidf_tsne,labels)
    print("running preprocessed eda")
