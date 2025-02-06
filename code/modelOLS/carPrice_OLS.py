import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# Reload data from CSV files
train_data = pd.read_csv("/tmp/AI-CPS/data/train/training_data.csv")
test_data = pd.read_csv("/tmp/AI-CPS/data/validation/test_data.csv")

# Split back into features and target
X_train = train_data.drop("price", axis=1)
y_train = train_data["price"]
X_test = test_data.drop("price", axis=1)
y_test = test_data["price"]

# ############################################# OLS MODEL #########################################
import statsmodels.api as sm
# Add constant to the model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# OLS model
ols_model = sm.OLS(y_train, X_train).fit()
print(ols_model.summary())

# Predict on test set
Y_pred_ols = ols_model.predict(X_test)

# Save the model
joblib.dump(ols_model, '/tmp/AI-CPS/data/learningBase_OLS/currentOlsSolution.pkl')

# ############################################# End of OLS MODEL #########################################

# ############################################# OLS Plots #########################################
with PdfPages("/tmp/AI-CPS/documentation/ols_model_plots.pdf") as pdf:
    # Scatter plot for OLS model
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, Y_pred_ols, label="OLS Predictions", alpha=0.6, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("OLS Model Predictions vs Actual")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Box plot for OLS model
    plt.figure(figsize=(12, 8))
    plt.boxplot([y_test, Y_pred_ols], labels=["Actual Price", "OLS Predictions"])
    plt.ylabel("Price")
    plt.title("Box Plot for OLS Model")
    pdf.savefig()
    plt.close()

    # Residual plot for OLS model
    residuals_ols = y_test - Y_pred_ols
    plt.figure(figsize=(12, 8))
    plt.scatter(Y_pred_ols, residuals_ols, label="OLS Residuals", alpha=0.6, color='red')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot for OLS Model")
    plt.legend()
    pdf.savefig()
    plt.close()

print("OLS model plots saved in 'OLS_Plots.pdf'")
# ############################################# End of OLS Plots #########################################
