# Midterm HW - CS 3120 - Machine Learning
# Author: Austin Gailey
# Instructor: Feng Jiang
import pandas as pd
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier


model1 = svm.SVC(kernel='poly', C=1.0,gamma='auto')
model2 = svm.SVC(kernel='linear', C=1.0,gamma='auto')
model3 = LogisticRegression()
model4 = DecisionTreeClassifier()
model5 = KNeighborsClassifier(n_neighbors=7)

models = [model1, model2, model3, model4, model5]

col_names    = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
feature_cols = ["sepal.length","sepal.width","petal.length","petal.width"]

iris = pd.read_csv("downloads/iris.csv", header=1, names=col_names)

print("Columns Selected: ")
print(feature_cols)

X = iris[feature_cols]
y = iris.variety

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=0)

for model in models:
    print("\n\nUsing Model: ", model,"\n")
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
print("Based on the classifiers used, The KNeighbors Classifier offers the best results.")
print("This could easily be best with another classifier given the proper parameters.")
print("The dataset is quite small so this analysis is also only best for our small dataset.")
print("Other factors which influence our result include, data split ratio, ultilizing validation hyperparamters or selecting specific columns.")