import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pandas as pd
from collections import Counter
from pandas import *
import seaborn as sns
import math

from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(stop_words='english', analyzer="word", use_idf=True, min_df=1, smooth_idf=True, norm='')
base = pd.read_csv("films.csv")
# pridanie novych prazdnych stlpcov z tfidf
base['tfidf1'] = 0
base['tfidf2'] = 0
base['tfidf3'] = 0
base['tfidf4'] = 0

# pocitanie idf
x = v.fit_transform(base.loc[: , 'storyline'].values.astype('U'))
idf = v.idf_
# zrobenie dictionary v tvare -> token : hodnota idf
dictineri = dict(zip(v.get_feature_names(), idf))

for i, row in base.iterrows():
    accStoryline = list(map(lambda x:x.lower(),row['storyline'].split()))
    trol = dict()
    # ulozenie hodnot tfidf s tokenmi do trola
    for accWord in accStoryline:
        foo = accWord.replace('.','')
        if foo in dictineri:
            if foo in trol:
                trol[foo] += dictineri[foo]
            else:
                trol[foo] = dictineri[foo]

    # normalizovanie trola podla logE
    for k, v in trol.items():
        trol[k] = int(round(math.log1p(v-1)))
    # spocitanie vyskytov jednotlivych normalizovanych tfidf pre dany dokument
    result = [0,0,0,0,0]
    for k, v in trol.items():
        if v == 0:
            result[0] += 1
        if v == 1:
            result[1] += 1
        if v == 2:
            result[2] += 1
        if v == 3:
            result[3] += 1
        if v == 4:
            result[4] += 1

    # ulozenie vyskytov do stlpcov dataframu
    base.set_value(i, 'tfidf1', result[1])
    base.set_value(i, 'tfidf2', result[2])
    base.set_value(i, 'tfidf3', result[3])
    base.set_value(i, 'tfidf4', result[4])

base = base.drop('storyline', 1)
# vytvorenie noveho csv
base.to_csv('base.csv', index=False)