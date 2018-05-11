import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
from matplotlib import style
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

### NACITANIE DAT ###

## STANDARD ##
train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)


## FEATURE SELECTION ##
# vybratie konkretnych slpcov ktore boli vybrane feature selectom

# train = train[["gross", "duration", "noir", "yearPublished", "monthPublished", "dharma", "productionCorpsOthers",
#                     "contentRating", "language", "location", "tfidf1", "tfidf2"]]
# test = test[["gross", "duration", "noir", "yearPublished", "monthPublished", "dharma", "productionCorpsOthers",
#                     "contentRating", "language", "location", "tfidf1", "tfidf2"]]

## FEATURE SELECTION WITH PCA ##
# train = train[['gross','duration','noir','yearPublished','monthPublished','dharma','productionCorpsOthers','contentRating','language','location',
#              'pca0','pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10','pca11','pca12','pca13',
#              'pca14','pca15','pca16', 'pca17', 'pca18', 'pca19','pca20','pca21','pca22','pca23','pca24','pca25','pca26',
#              'pca27','pca28','pca29','pca30','pca31','pca32', 'pca33', 'pca34', 'pca35']]
# test = test[['gross','duration','noir','yearPublished','monthPublished','dharma','productionCorpsOthers','contentRating','language','location',
#              'pca0','pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10','pca11','pca12','pca13',
#              'pca14','pca15','pca16', 'pca17', 'pca18', 'pca19','pca20','pca21','pca22','pca23','pca24','pca25','pca26',
#              'pca27','pca28','pca29','pca30','pca31','pca32', 'pca33', 'pca34', 'pca35']]



### HEADERY ###

## train headers ##
train_headers = list(train.columns.values)
# numpy array pre sklearn
train_np = train.as_matrix()
train_X = train_np[:, 1:]  # select vsetky stlpce okrem prveho
train_y = train_np[:, 0]   # select prvy stlpec

## test headers ##
test_headers = list(test.columns.values)
# numpy array pre sklearn
test_np = test.as_matrix()

test_X = test_np[:, 1:]  # select vsetky stlpce okrem prveho
test_y = test_np[:, 0]   # select prvy stlpec


### Feature selection ####
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

m = RFECV(RandomForestClassifier(n_estimators=200, warm_start=True, criterion='gini', max_depth=13), scoring='accuracy')
fit = m.fit(train_X, train_y)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)

for x in range(0, 83):
    if fit.ranking_[x] == 1 :
        print(train_headers[x])


### Logistic regression ####
logisticRegr = LogisticRegression()
logisticRegr.fit(train_X, train_y)
logisticRegr.predict(test_X)
pred_y = logisticRegr.predict(test_X)
print('Logistic Regression')
print(f1_score(test_y, pred_y, average="macro"))
print(precision_score(test_y, pred_y, average="macro"))
print(recall_score(test_y, pred_y, average="macro"))


### Random forest with cross validation ####
print("Random forest with best setting")
x = 0
for y in range(0,10):
    clf = RandomForestClassifier(n_estimators=1000, warm_start=False, criterion='gini', max_depth=20, max_features=12, oob_score=True, class_weight='balanced')
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    x += f1_score(test_y, pred_y, average="macro")
    # print(f1_score(test_y, pred_y, average="macro"))
    # print("----------")

print("===============")
print(x/10)


### Naive Bayes ###
gnb = GaussianNB()
pred_y = gnb.fit(train_X, train_y).predict(test_X)
print('Gaussian NB')
print(f1_score(test_y, pred_y, average="macro"))
print(precision_score(test_y, pred_y, average="macro"))
print(recall_score(test_y, pred_y, average="macro"))


# SVM regularization parameter pre SVM klasifikatory
C = 1

# SVC with linear kernel #
svc = svm.SVC(kernel='linear', C=C).fit(train_X, train_y)
pred_y = svc.predict(test_X)
print('SVC with linear kernel')
print(f1_score(test_y, pred_y, average="macro"))
print(precision_score(test_y, pred_y, average="macro"))
print(recall_score(test_y, pred_y, average="macro"))

### LinearSVC (linear kernel) ####
svc = svm.LinearSVC(C=C).fit(train_X, train_y)
pred_y = svc.predict(test_X)
print('LinearSVC (linear kernel)')
print(f1_score(test_y, pred_y, average="macro"))
print(precision_score(test_y, pred_y, average="macro"))
print(recall_score(test_y, pred_y, average="macro"))

### SVC with RBF kernel ####
svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_X, train_y)
pred_y = svc.predict(test_X)
print('SVC with RBF kernel')
print(f1_score(test_y, pred_y, average="macro"))
print(precision_score(test_y, pred_y, average="macro"))
print(recall_score(test_y, pred_y, average="macro"))

#### SVC with polynomial (degree 3) kernel ####
svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_X, train_y)
pred_y = svc.predict(test_X)
print('SVC with polynomial (degree 3) kernel')
print(f1_score(test_y, pred_y, average="macro"))
print(precision_score(test_y, pred_y, average="macro"))
print(recall_score(test_y, pred_y, average="macro"))


### VYGENEROVANIE NORMALIZOVANEJ HEATMAPY PODLA KATEGORII
def percenta(x, sum):
    return float((x/sum))

cm = confusion_matrix(test_y, pred_y)

# spocitanie vsetkych predikovanych filmov skrz kategorie
print(cm)
sum = [0 for x in range(0,12)]
foo = 0
for row in cm:
    for val in row:
        sum[foo] += val
    foo += 1
print(sum)

# prepocitanie na percenta a vlozenie do CM2
foo = 0
cm2 = [[0 for x in range(0,12)] for y in range(0,12)]
for i in range(0, 12):
    for j in range(0, 12):
        cm2[i][j] = percenta(float(cm[i][j]), sum[foo])
    foo += 1
print(cm2)

style.use('ggplot')
sns.heatmap(cm2, cmap="YlGnBu", annot=True, fmt='.1%')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.subplot().xaxis.set_ticklabels([8,9,10,11,12,13,14,15,16,17,18,19])
plt.subplot().yaxis.set_ticklabels([8,9,10,11,12,13,14,15,16,17,18,19])
plt.show()
