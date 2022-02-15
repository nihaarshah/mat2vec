"""
This script does the following:
Loads various word2vec models with different hyperparameters, then obtains the word embeddings for common words
in its vocabulary (>5 frequency). Then performs KMeans clustering with N=3 clusters on the word vectors.
Finally performs PCA (Principal Component Analysis) on the word vectors and projects them into a 2D plane so we
can visualize the plots. The plots come with color coded words too indicating the cluster assigned to that word.
------------------------------------------------------------NOTE--------------------------------------------------
Essentially tasks that were to be done in mat_sensitivity.py, mat_plot.py and mat_cool.py are all done in one script.
@ Author: Nihaar Shah
"""
import gensim
from gensim.models import Word2Vec
import warnings
import pickle
from nltk.cluster import KMeansClusterer
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = [r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\default_size",
    r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\size_100",
    r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\size_300",
    r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\size_400",
    r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\window_5",
    r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\window_10",
    r"C:\Users\212803971\Documents\A-course\software_proj\fresh_start\mat2vec\mat2vec\training\models\other_models\window_15"]


warnings.simplefilter("ignore", np.ComplexWarning)

def plot_word_embeddings(model):

    print("The model vocab is as follows ---------------------------------------------")
    print(list(model.wv.vocab))


    X = model[model.wv.vocab]
    NUM_CLUSTERS=3
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    print (assigned_clusters)
    labels = np.array(assigned_clusters)
    assigned_0 = np.where(labels==0,1,0)
    assigned_1 = np.where(labels==1,1,0)
    assigned_2 = np.where(labels==2,1,0)
    # PCA
    df =pd.DataFrame(X)
    #Computing the correlation matrix
    X_corr=df.corr()

    #Computing eigen values and eigen vectors
    values,vectors=np.linalg.eig(X_corr)

    #Sorting the eigen vectors coresponding to eigen values in descending order
    args = (-values).argsort()
    values = vectors[args]
    vectors = vectors[:, args]

    #Taking first 2 components which explain maximum variance for projecting
    new_vectors=vectors[:,:2]

    #Projecting it onto new dimesion with 2 axis
    neww_X=np.dot(X,new_vectors)


    neww_X_df = pd.DataFrame(neww_X)
    filtered_label0 = neww_X_df[assigned_0 == 1]
    filtered_label1 = neww_X_df[assigned_1 == 1]
    filtered_label2 = neww_X_df[assigned_2 == 1]

    plt.figure()
    # plotting the results
    plt.scatter(filtered_label0.loc[:, 0], filtered_label0.loc[:, 1],color='red')
    plt.scatter(filtered_label1.loc[:, 0], filtered_label1.loc[:, 1],color='black')
    plt.scatter(filtered_label2.loc[:, 0], filtered_label2.loc[:, 1],color='blue')

    plt.xlabel("PC1",size=15)
    plt.ylabel("PC2",size=15)

    plt.title("Word Embedding Space: %s"%(model_name.split('\\')[-1]),size=20)
    vocab=list(model.wv.vocab)
    for i, word in enumerate(vocab):
      plt.annotate(word,xy=(neww_X[i,0],neww_X[i,1]))

    plt.savefig("%s.png"%(model_name.split('\\')[-1]))
    # plt.show()

for model_name in s:
    model = Word2Vec.load(model_name)
    plot_word_embeddings(model)




