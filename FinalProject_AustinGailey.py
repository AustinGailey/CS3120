# Final Project - Basic Neural Network
# Author: Austin Gailey
# Professor: Feng Jiang
# Course: CS3120 - Machine Learning
# Date: 13 May 2021
#
# Description:  This neural network will vectorize Amazon reviews (How NLP data is processed), 
#      then classify whether the rating is greater or lesser than 2.5 stars.
#
# Requirements:
#   Note: All Requirements can be installed using pip install [Name of Package]
#   Packages:
#   1. matplotlib
#   2. numpy
#   3. pandas
#   4. sklearn
#   5. ntlk
#   6. tensorflow
#
#   Dataset: 
#      Parent: https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
#      Direct Link:  https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz
#           (Don't forget to unzip!)

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import nltk
import tensorflow
import keras.losses
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

dataset_path = "~/downloads/amazon_reviews_us_Video_Games_v1_00.tsv"
col_names    = ["marketplace","customer_id","review_id","product_id", "product_parent","product_title","product_category","star_rating","helpful_votes","total_votes","vine","verified_purchase","review_headline","review_body","review_date"]

print("Reading Input Data and Removing Lines With Errors...  ")
vg = pd.read_csv(dataset_path, header=1, names=col_names,sep="\t",error_bad_lines=False)
vg.dropna(inplace=True) # Drops Data with NaN values
print("Finisehd Reading Data.")

nltk.download('stopwords') # Downloads List of Stopwords
swords = set(stopwords.words('english')) # Set nltk Stopword List Equal to a Variable

# Set Feature and Target Values
X = vg['review_body']
y_source = vg['star_rating']
y = y_source > 2.5  # Converts Integers into Booleans Greater or Lesser than 2.5

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=0) # Splits Data

# Vectorize Words for NLP ML
print("Beginning Vectorization... This make take a few minutes...")
cvec=CountVectorizer(stop_words=swords, max_df=.20, min_df=3, max_features=500)
X_train_sc = cvec.fit_transform(X_train)
X_test_sc = cvec.transform(X_test)
print("Vectorization Complete!")

# Formats Vectorized Data
X_train_sc = pd.DataFrame.sparse.from_spmatrix(X_train_sc)
X_test_sc = pd.DataFrame.sparse.from_spmatrix(X_test_sc)

n_input = X_train_sc.shape[1] # Sets Input Shape for Training

# Begin Building the Model
model = Sequential()
model.add(Dense(
128, # Number of Nodes
input_dim = n_input, # Input Dimension (Number of Features)
activation = 'relu' # Applies the Rectified Linear Unit Activation Function. (Sigmoid is another option)
))

model.add(Dense(64, activation='relu')) # Hidden Layer

model.add(Dense(1, activation='sigmoid')) # Classification (Final Layer)

loss_fn = keras.losses.BinaryCrossentropy()

# Compile Model
print("Begin Training...")
model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['accuracy'])

# Fit Model on Training Data - This runs for a While.  (Grab a snack or reduce the number of epochs)
# Stores Results in History
history = model.fit(X_train_sc,
                    y_train, 
                    #batch_size=256,   #Can Help Reduce Memory Requirements
                    validation_data=(X_test_sc, y_test),
                    epochs=20, #Will Run 'N' Times, Refining Model During Each Epoch
                    verbose=1)
print("Training Complete!")

# Visualize Loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']

plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='Training loss', color='navy')
plt.plot(test_loss, label='Testing loss', color='skyblue')
plt.title('NN Loss', size=30)
plt.legend()