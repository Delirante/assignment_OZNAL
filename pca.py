import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
import pandas as pd
import numpy as np

n = 36
base = pd.read_csv("films.csv")
for x in range(0, n):
    name = "pca"+str(x)
    base[name] = 0

# calculate tf-idf of texts
tf_idf_vectorizer = TfidfVectorizer(stop_words='english', analyzer="word", use_idf=True, min_df=1, smooth_idf=True, norm='')
tf_idf_matrix = tf_idf_vectorizer.fit_transform(base.loc[: , 'storyline'].values.astype('U'))

trol = TruncatedSVD(n_components=n)
reduced_data = trol.fit_transform(tf_idf_matrix)
print("TruncatedSVD "+str(n))
print(sum(trol.explained_variance_))

shape = reduced_data.shape
result = np.zeros(shape)
for x in range(0, shape[0]):
    for y in range(0, shape[1]):
        foo = int(float(reduced_data[x,y]) * 100)
        base.set_value(x, 'pca'+str(y), foo)

base = base.drop('storyline', 1)
# vytvorenie noveho csv
base.to_csv('basePCA.csv', index=False)
