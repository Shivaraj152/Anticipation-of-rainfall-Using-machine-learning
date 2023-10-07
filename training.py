import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings(action='ignore')

data = pd.read_csv("fall.csv")
print("Successfully Imported Data!")
data.head()
data.info()
print(data.shape)

data.describe(include='all')

print(data.isna().sum())
data.corr()

data.groupby('Rainfall').mean()

sns.countplot(data['Rainfall'])
plt.show()




corr = data.corr()
sns.heatmap(corr,annot=True)
plt.show()




X = data.drop('year',1)
y = data['Rainfall']

# split the data train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)

from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LogisticRegression
X = data.drop('year',1)
y = data['Rainfall']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
# making predictions on the testing set
y_pred = model.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn.metrics import accuracy_score
acclr = metrics.accuracy_score(y_test, y_pred)*200
print("Linear Regression model accuracy(in %):",acclr)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
test_targets=y_test
cm = confusion_matrix(test_targets,y_pred)

from sklearn.metrics import roc_curve
fpr ,tn, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
Sensitivity= tn / (tn+fpr)

print('Sensitivity='+str(Sensitivity[1]))
precisioncnn = precision_score((test_targets)>=4,(y_pred)>=1)

print('precision='+str(precisioncnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
f1scorecnn = f1_score((test_targets)>=4,(y_pred)>=1)

print('f1-score='+str(f1scorecnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
recallscorecnn = recall_score((test_targets)>=4,(y_pred)>=1)

print('recall-score='+str(recallscorecnn))
import pickle
file = 'finalized_model_LR.sav'
pickle.dump(model, open(file, 'wb'))
###random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clfRF = RandomForestClassifier(max_depth=2, random_state=0)
clfRF.fit(X_train, y_train)
y_pred = clfRF.predict(X_test)

test_targets=y_test
#y_pred=model.predict(train_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
cm = confusion_matrix(test_targets,y_pred)

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve((test_targets)>=1,(y_pred)>=1)

accuracyrf = (accuracy_score((test_targets)>=4,(y_pred)>=1)*100)


print(cm)


print('RF Accuracy='+str(accuracyrf))
fpr ,tn, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
Sensitivity= tn / (tn+fpr)

print('Sensitivity='+str(Sensitivity[1]))
precisioncnn = precision_score((test_targets)>=4,(y_pred)>=1)

print('precision='+str(precisioncnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
f1scorecnn = f1_score((test_targets)>=4,(y_pred)>=1)

print('f1-score='+str(f1scorecnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=4,(y_pred)>=1)
recallscorecnn = recall_score((test_targets)>=4,(y_pred)>=1)

print('recall-score='+str(recallscorecnn))

import pickle
file1 = 'finalized_model_RF.sav'
pickle.dump(clfRF, open(file1, 'wb'))
##decision tree
from sklearn import tree

clfDT = tree.DecisionTreeClassifier()
clfDT = clfDT.fit(X_train, y_train)

y_pred = clfDT.predict(X_test)


test_targets=y_test
#y_pred=model.predict(train_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
cm = confusion_matrix(test_targets,y_pred)

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve((test_targets)>=1,(y_pred)>=1)

accuracydt = (accuracy_score((test_targets)>=3,(y_pred)>=1)*100)


print(cm)


print('DT Accuracy='+str(accuracydt))
fpr ,tn, thresholds = roc_curve((test_targets)>=3,(y_pred)>=1)
Sensitivity= tn / (tn+fpr)

print('Sensitivity='+str(Sensitivity[1]))
precisioncnn = precision_score((test_targets)>=3,(y_pred)>=1)

print('precision='+str(precisioncnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=3,(y_pred)>=1)
f1scorecnn = f1_score((test_targets)>=3,(y_pred)>=1)

print('f1-score='+str(f1scorecnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=3,(y_pred)>=1)
recallscorecnn = recall_score((test_targets)>=3,(y_pred)>=1)

print('recall-score='+str(recallscorecnn))

import pickle
file2 = 'finalized_model_DT.sav'
pickle.dump(clfDT, open(file2, 'wb'))

#navie bayes
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
y_pred = NB.fit(X_train, y_train)

y_pred= NB.predict(X_test)

print('#########################################################')

test_targets=y_test
#y_pred=model.predict(train_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score
cm = confusion_matrix(test_targets,y_pred)

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve((test_targets)>=1,(y_pred)>=1)

accuracynb = (accuracy_score((test_targets)>=2,(y_pred)>=1)*100)


print(cm)


print('NB Accuracy='+str(accuracynb))
fpr ,tn, thresholds = roc_curve((test_targets)>=2,(y_pred)>=1)
Sensitivity= tn / (tn+fpr)

print('Sensitivity='+str(Sensitivity[1]))
precisioncnn = precision_score((test_targets)>=2,(y_pred)>=1)

print('precision='+str(precisioncnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=2,(y_pred)>=1)
f1scorecnn = f1_score((test_targets)>=2,(y_pred)>=1)

print('f1-score='+str(f1scorecnn))

fpr ,tpr, thresholds = roc_curve((test_targets)>=2,(y_pred)>=1)
recallscorecnn = recall_score((test_targets)>=3,(y_pred)>=1)

print('recall-score='+str(recallscorecnn))


display = confusion_matrix(y_test, y_pred)
print(display)

import pickle
file3 = 'finalized_model_NB.sav'
pickle.dump(NB, open(file3, 'wb'))

import matplotlib.pyplot as plt
x=['logistic regression', 'RF', 'DT', 'NB']
y=[acclr,accuracyrf,accuracydt,accuracynb]
plt.bar(x,y)
plt.xlabel('Algorithm')
plt.ylabel("Accuracy")
plt.title('Accuracy Bar Plot')
plt.show()
