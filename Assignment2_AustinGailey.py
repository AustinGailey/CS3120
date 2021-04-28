# HW 2
# Austin Gailey
# 03/16/21
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import classification_report


# Data Column Names
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# Load Dataset
pima = pd.read_csv("downloads/pima-indians-diabetes.csv", header=None, names=col_names)

# Selected Columns
feature_cols = ['glucose', 'skin', 'insulin', 'bmi', 'age']

print("Columns Selected: ")
print(feature_cols)

X = pima[feature_cols]
y = pima.label

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.4, random_state=0)


# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# Fit model
logreg.fit(X_train,y_train)

# Y Score
y_score = logreg.decision_function(X_test)

# Predict Y values
y_pred = logreg.predict(X_test)


# Confusion Matrix Display
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

# Classification Report
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Receiver Operating Characteristic
fpr, tpr, thresholds = roc_curve(y_test, y_score)
#ROC AUC Score
roc_auc = auc(fpr, tpr)

#ROC Curve
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC Curve")
display.plot()
plt.show()    