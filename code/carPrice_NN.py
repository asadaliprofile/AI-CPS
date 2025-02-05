#import paskages 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import *
from keras import backend as K
import csv
import random
from matplotlib.backends.backend_pdf import PdfPages


#Load dataset car
df_cars= pd.read_csv("/Users/Asad Ali/AI-CPS/data/joint_data_collection.csv")

#print head data
df_cars.head()

#name columns
df_cars.columns
#print describtion dataset 
df_cars.describe()
#print information data
df_cars.info()
#check is null data
df_cars.isnull().sum()

#get columns type is object 
categorical_cols =df_cars.dtypes[df_cars.dtypes=="object"].index.to_list()
categorical_cols

# Encoding categorical variables 
label_encoder = LabelEncoder()

for col_cat in categorical_cols:
    df_cars[col_cat]= label_encoder.fit_transform(df_cars[col_cat])
    print("Apply label encode for column ",col_cat)

#get columns type is numric  
numerical_cols  = df_cars.dtypes[df_cars.dtypes!="object"].index.to_list()
numerical_cols 

#but exception price 
numerical_cols.remove('price')
numerical_cols

# Display summary statistics of numerical features  mean, standard deviation, minimum, and maximum 
df_cars[numerical_cols].describe().loc[['mean', 'std','min','max']]


#Standardizes features by scaling each feature to a given range
minmax =MinMaxScaler()#MinMaxScaler
df_cars[numerical_cols] = minmax.fit_transform(df_cars[numerical_cols])


# Split data into features and target
X=df_cars.drop("price" , axis=1)    
y=df_cars['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)


# Combine features and target for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save to CSV files
train_data.to_csv("/Users/Asad Ali/AI-CPS/data/training_data.csv", index=False)
test_data.to_csv("/Users/Asad Ali/AI-CPS/data/test_data.csv", index=False)


# Load data from CSV files
train_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/training_data.csv")
test_data = pd.read_csv("/Users/Asad Ali/AI-CPS/data/test_data.csv")


# Save file activation_data.csv
def select_row(df, output_file):
    header = df.columns.tolist()  # Store the header
    random_row = df.sample(n=1).iloc[0].tolist()  # Select a random row
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header first
        writer.writerow(random_row)  # Write the selected row
    
    print(f"5th row saved to {output_file}")
select_row(test_data, "/Users/Asad Ali/AI-CPS/data/activation_data.csv")



# Split back into features and target
X_train = train_data.drop("price", axis=1)
y_train = train_data["price"]
X_test = test_data.drop("price", axis=1)
y_test = test_data["price"]


# Print the shapes of the training and testing sets
print("Training set (features):", X_train.shape)
print("Testing set (features):", X_test.shape)
print("Training set (target):", y_train.shape)
print("Testing set (target):", y_test.shape)



# Define a custom callback to record R^2 score
class R2Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(R2Callback, self).__init__()
        self.validation_data = validation_data
        self.r2_values = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        r2 = r_squared(y_val, y_pred)
        self.r2_values.append(r2)
        print(f' - val_r2: {r2:.4f}')
# Compute R^2 score using TensorFlow operations
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res/(SS_tot + K.epsilon())


# Build model
model = tf.keras.models.Sequential([
    # Input layer with 1024 neurons and ReLU activation function
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)), 
    # Hidden layers with ReLU activation functions
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(256, activation='relu'), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.Dense(8, activation='relu'), 
    tf.keras.layers.Dense(4, activation='relu'), 
    # Output Layer
    tf.keras.layers.Dense(1)  
])

# Compile your model with the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_weights.weights.h5', monitor='val_loss', mode='min',
    save_best_only=True, save_weights_only=True, verbose=1)

# Train your model
history = model.fit(X_train, y_train, epochs=300, batch_size=45, validation_split=0.2, 
                    callbacks=[reduce_lr, early_stopping, model_checkpoint])


# Save the trained model
model.save("/Users/Asad Ali/AI-CPS/data/currentAiSolution.h5")

y_pred_nn = model.predict(X_test).flatten()

# ############################################# Plots #########################################
with PdfPages("/Users/Asad Ali/AI-CPS/documentation/NN_Model_Plots.pdf") as pdf:
    # Plot number of iterations for NN model training
    plt.figure(figsize=(12, 8))
    plt.plot(history.epoch, history.history["loss"], label="Training Loss", color='blue')
    plt.plot(history.epoch, history.history["val_loss"], label="Validation Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network Training Loss Over Iterations")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Plot training history
    plt.figure(figsize=(12, 6))
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='train_mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.legend()

    # Plot MSE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mse'], label='train_mse')
    plt.plot(history.history['val_mse'], label='val_mse')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error (MSE)')
    plt.legend()
    pdf.savefig()
    plt.close()


    # Scatter plot for Neural Network model
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred_nn, label="Neural Network Predictions", alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Neural Network Model Predictions vs Actual")
    plt.legend()
    pdf.savefig()
    plt.close()

    # Box plot for Neural Network model
    plt.figure(figsize=(12, 8))
    plt.boxplot([y_test, y_pred_nn], labels=["Actual Price", "Neural Network Predictions"])
    plt.ylabel("Price")
    plt.title("Box Plot for Neural Network Model")
    pdf.savefig()
    plt.close()

    # Residual plot for Neural Network model
    residuals_nn = y_test - y_pred_nn
    plt.figure(figsize=(12, 8))
    plt.scatter(y_pred_nn, residuals_nn, label="Neural Network Residuals", alpha=0.6, color='blue')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot for Neural Network Model")
    plt.legend()
    pdf.savefig()
    plt.close()


    # Histogram of Neural Network Residuals
    plt.figure(figsize=(12, 8))
    plt.hist(residuals_nn, bins=30, color='blue', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Neural Network Residuals")
    pdf.savefig()
    plt.close()

print("Neural Network model plots saved in 'NN_Model_Plots.pdf'")
# ############################################# End of Plots #########################################