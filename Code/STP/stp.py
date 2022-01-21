import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

glassdata = pd.read_csv('G:\glasss\glass.csv')

np.unique(glassdata['Type'])

types = glassdata.Type.unique()

alpha = 0.7

train = pd.DataFrame()
test = pd.DataFrame()
for i in range(len(types)):
    tempt = glassdata[glassdata.Type == types[i]]
    train = train.append(tempt[0:int(alpha*len(tempt))])
    test = test.append(tempt[int(alpha*len(tempt)): len(tempt)])

print (train.shape, test.shape, glassdata.shape)

#train.describe()

print ((train['Ba']==0).sum()/len(train))
print ((train['Fe']==0).sum()/len(train))

train_variables = train.drop(['Type','Ba', 'Fe'],1)
train_variable_corrmat = train_variables.corr()
#print (train_variable_corrmat)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,8))
sns.pairplot(train_variables,palette='coolwarm')
corr = train_variables.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar = True,  square = True, annot=True,
           xticklabels= corr.columns.values, yticklabels= corr.columns.values,
           cmap= 'coolwarm')
plt.show()

X = train.drop('Type',1)
Y = train['Type']
Z = test.drop('Type',1)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

prediction = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(Z)


truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction == truth)/(len(truth))

X1 = train[['Al', 'Si']]
Y1 = train['Type']
Z1 = test[['Al', 'Si']]


prediction1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X1, Y1).predict(Z1)


truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction1 == truth)/(len(truth))


types = np.unique(train['Type'])


X1 = train[['Al', 'K', 'Mg']]
Y1 = train['Type']
Z1 = test[['Al', 'K', 'Mg']]


prediction1 = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X1, Y1).predict(Z1)


truth = test['Type']
truth = np.array(truth)

accuracy = sum(prediction1 == truth)/(len(truth))
print ("Accuracy =", accuracy)
print ("Predicted Values =", prediction1)
print ("Actual Values=", truth)
cnf_matrix=confusion_matrix(truth,prediction1)
print(cnf_matrix)
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
print("False Positives =",FP)
print("False Negatives =",FN)
print("True Positives =",TP)
print("True Negatives =",TN)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

print("Sensitivity =",TPR)
print("Specificity =",TNR)
print("Precision =",PPV)
print("False Positive Rate =",FPR)
print("False Negative Rate =",FNR)

