import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pandas as pd
from collections import Counter
from pandas import *
import seaborn as sns
import math
from random import *

## STANDARD
base = pd.read_csv("base.csv")
base = base[['gross','id','duration','Rating','action','adventure','biography','comedy','crime','drama','family',
             'fantasy','history','horror','mystery','romance','scifi','thriller','war','western','musical','animation',
             'sport','music','noir','yearPublished','monthPublished','budget','disney','warner','sony','universal',
             'fox','paramount','summit','dimension','mgm','dreamworks','marvel','miramax','columbia','united','utv',
             'pixomondo','dharma','productionCorpsOthers','contentRating','language','location',
             'tfidf1','tfidf2','tfidf3','tfidf4']]

## PCA
# base = pd.read_csv("basePCA.csv")
# base = base[['gross','id','duration','Rating','action','adventure','biography','comedy','crime','drama','family',
#              'fantasy','history','horror','mystery','romance','scifi','thriller','war','western','musical','animation',
#              'sport','music','noir','yearPublished','monthPublished','budget','disney','warner','sony','universal',
#              'fox','paramount','summit','dimension','mgm','dreamworks','marvel','miramax','columbia','united','utv',
#              'pixomondo','dharma','productionCorpsOthers','contentRating','language','location',
#              'pca0','pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10','pca11','pca12','pca13',
#              'pca14','pca15','pca16', 'pca17', 'pca18', 'pca19','pca20','pca21','pca22','pca23','pca24','pca25','pca26',
#              'pca27','pca28','pca29','pca30','pca31','pca32', 'pca33', 'pca34', 'pca35']]



# odstranenie ID lebo toto je nanic pre clasifikator
base = base.drop('id', 1)

# vytvorenie tvrdej kopie datasetu
train = base.copy(deep=True)
# odstranenie riadkov z testu
test = base[0:0]

# pocet zaznamov v datasete
max = 11452
# for cyklus pre vyber 10%
for x in range(0, 1145):
    random = randint(1, max)
    # prilepenie noveho riadku do testu
    test = test.append(train.iloc[[random]])
    # odstranenie riadku z trainu
    train = train.drop(train.index[[random]])
    # novy max sa musi znizit o jedna lebo aj pocet zaznamov v traine sa nam znizil o jedna
    max = max - 1


# vytvorenie noveho csv
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)