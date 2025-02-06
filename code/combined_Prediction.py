#import paskages 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import statsmodels.api as sm

train_data = pd.read_csv("/tmp/AI-CPS/data/train/training_data.csv")
test_data = pd.read_csv("/tmp/AI-CPS/data/validation/test_data.csv")
activation_data = pd.read_csv("/tmp/AI-CPS/data/activationBase/activation_data.csv")
# Split back into features and target
X_train = train_data.drop("price", axis=1)
y_train = train_data["price"]
X_test = test_data.drop("price", axis=1)
y_test = test_data["price"]

# ############################################# NN model Predictions #########################################
model = keras.models.load_model('/tmp/AI-CPS/data/learningBase_NN/currentAiSolution.h5')
# Print model summary to verify
model.summary()
y_pred_nn = model.predict(X_test).flatten()

activation_test = activation_data.drop("price", axis=1)
one_pred = model.predict(activation_test).flatten()
print("Predicted from activation_data.csv", one_pred)
# ############################################# End of NN model Predictions #########################################

# ############################################# OLS Model Predictions #########################################
X_test = sm.add_constant(X_test)
# Load the saved model
loaded_model = joblib.load('/tmp/AI-CPS/data/learningBase_OLS/currentOlsSolution.pkl')

# Make predictions
Y_pred_ols = loaded_model.predict(X_test)

# Save predictions along with actual values
results = pd.DataFrame({"Actual Price": y_test, "Predicted Price": Y_pred_ols})
results.to_csv("ols_predictions.csv", index=False)
print("Predicted values:", Y_pred_ols)
# ############################################# End of OLS model Predictions #########################################



# ############################################# Combined Plots #########################################

with PdfPages("/tmp/AI-CPS/documentation/comparison_plots.pdf") as pdf:
    # Scatter plot comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_nn, label="Neural Network Predictions", alpha=0.6, color='blue')
    plt.scatter(y_test, Y_pred_ols, label="OLS Predictions", alpha=0.6, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Comparison of Neural Network and OLS Model Predictions")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Box plot comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([y_test, y_pred_nn, Y_pred_ols], labels=["Actual Price", "Neural Network Predictions", "OLS Predictions"])
    plt.ylabel("Price")
    plt.title("Box Plot Comparison of Actual and Predicted Prices")
    pdf.savefig()
    plt.close()
# ############################################# End of Combined Plots #########################################