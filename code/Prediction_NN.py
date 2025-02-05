#import paskages 
import pandas as pd
from keras.layers import *
from tensorflow import keras
from keras import backend as K

train_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/training_data.csv")
test_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/test_data.csv")
activation_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/activation_data.csv")
# Split back into features and target
X_train = train_data.drop("price", axis=1)
y_train = train_data["price"]
X_test = test_data.drop("price", axis=1)
y_test = test_data["price"]

# ############################################# NN model Predictions #########################################
model = keras.models.load_model('/Users/Asad Ali/AI-CPS/data/currentAiSolution.h5')
# Print model summary to verify
model.summary()
y_pred_nn = model.predict(X_test).flatten()

activation_test = activation_data.drop("price", axis=1)
one_pred = model.predict(activation_test).flatten()
print("Predicted from activation_data.csv", one_pred)