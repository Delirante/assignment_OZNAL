import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pandas as pd
from collections import Counter
from pandas import *
import seaborn as sns

# Content rating
style.use('ggplot')
data3 = pd.read_csv("films.csv")
c = Counter(data3.loc[: , 'contentRating'])
plt.bar(c.keys(), c.values(), color='b')
plt.yticks(range(0, 5500, 500))
plt.xticks(range(0, 11, 1))
plt.title("Histogram - Content rating")
plt.show()

# Rating
style.use('ggplot')
data3 = pd.read_csv("films.csv")
c = Counter(data3.loc[: , 'Rating'])
plt.bar(c.keys(), c.values(), color='b')
plt.yticks(range(0, 550, 50))
plt.xticks(range(0, 110, 10))
plt.title("Histogram - Rating")
plt.show()


# Location
style.use('ggplot')
data3 = pd.read_csv("films.csv")
c = Counter(data3.loc[: , 'location'])
plt.bar(c.keys(), c.values(), color='b')
plt.yticks(range(0, 6500, 500))
plt.xticks(range(0, 11, 1))
plt.title("Histogram - Location")
plt.show()


# Language
style.use('ggplot')
data3 = pd.read_csv("films.csv")
c = Counter(data3.loc[: , 'language'])
plt.bar(c.keys(), c.values(), color='b')
plt.yticks(range(0, 11000, 1000))
plt.xticks(range(0, 11, 1))
plt.title("Histogram - Language")
plt.show()

# Budget
style.use('ggplot')
data3 = pd.read_csv("films.csv")
c = Counter(data3.loc[: , 'budget'])
plt.bar(c.keys(), c.values(), color='b')
plt.yticks(range(0, 6500, 500))
plt.xticks(range(12, 20, 1))
plt.title("Histogram - Budget")
plt.show()

# Gross
style.use('ggplot')
data3 = pd.read_csv("films.csv")
c = Counter(data3.loc[: , 'gross'])
plt.bar(c.keys(), c.values(), color='b')
plt.yticks(range(0, 1900, 100))
plt.xticks(range(8, 20, 1))
plt.title("Histogram - Gross")
plt.show()


# HeatMap date
style.use('ggplot')
data3 = pd.read_csv("yearsmonths.csv")
trol = data3.pivot(index='yearPublished', columns='monthPublished', values='count')
sns.heatmap(trol, cmap="YlGnBu")
plt.title("Heatmap - vydanie filmov skrz roky a mesiace")
plt.show()


# Vysledky klasifikacie
d = {}
d["Precision"] = 30.83
d["Recall"] = 28.64
d["F1 score"] = 28.38
style.use('ggplot')
colormap = np.array(['r', 'g', 'b'])
plt.bar(d.keys(), d.values(), color=colormap)
plt.title("Random Forest")
plt.yticks(range(0, 36, 2))
plt.show()
