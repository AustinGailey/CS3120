# HW 3
# Austin Gailey
# 03/20/21
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imutils import paths
import numpy as np
import scipy.misc
import cv2
import os

def load(imagePath_list, verbose=-1):
        data = []
        labels = []
        for (i, imagePath) in enumerate(imagePath_list):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            image = cv2.resize(image, (32, 32),interpolation=cv2.INTER_CUBIC)
            data.append(image)
            labels.append(label)
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePath_list)))
                return (np.array(data), np.array(labels))


print("[INFO] loading images...")
imagePath_list = list(paths.list_images("downloads/KNN/animals"))
(data, labels) = load(imagePath_list, verbose=3000)
data = data.reshape((data.shape[0], 3072))
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))      
data = np.array(data)
labels = np.array(labels)
data = data.reshape((data.shape[0], 3072))
le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.30, random_state=0)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.66, random_state=0)

k_values = [3, 5, 7]
distance_metrics = {1: "Manhattan", 2: "Euclidean"}
best_accuracy = 0
best_params = []

for l, type in distance_metrics.items():
    print("\n\nL-Type:", type, "\n")
    for k in k_values:
        print("K-Value:", k)
        model = KNeighborsClassifier(n_neighbors=k, p = l)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_valid)
        tested_accuracy = accuracy_score(y_valid, y_pred)
        print("Accuracy:", tested_accuracy, "\n")
        
        if tested_accuracy > best_accuracy:
            best_accuracy = tested_accuracy
            best_params = [k, l]  

print("\nBest combo:", "K Value = " + str(best_params[0]), 
      "and Distance Metric = " +  distance_metrics[best_params[1]])

print("\nGenerating Classification Report...")
model = KNeighborsClassifier(n_neighbors=best_params[0], p = best_params[1])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))