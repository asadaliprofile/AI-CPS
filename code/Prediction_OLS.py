#import paskages 
import pandas as pd
import joblib
import statsmodels.api as sm

train_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/training_data.csv")
test_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/test_data.csv")
activation_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/activation_data.csv")
# Split back into features and target
X_train = train_data.drop("price", axis=1)
y_train = train_data["price"]
X_test = test_data.drop("price", axis=1)
y_test = test_data["price"]

# ############################################# OLS Model Predictions #########################################
X_test = sm.add_constant(X_test)
# Load the saved model
loaded_model = joblib.load('/Users/Asad Ali/AI-CPS/data/currentOlsSolution.pkl')

# Make predictions
activation_test = activation_data.drop("price", axis=1)
activation_test.insert(0, 'Intercept', 1)
Y_pred_ols = loaded_model.predict(activation_test)

results = pd.DataFrame({"Actual Price": y_test, "Predicted Price": Y_pred_ols})
# results.to_csv("ols_predictions.csv", index=False)
print("Predicted value:", Y_pred_ols)
# ############################################# End of OLS model Predictions #########################################

