# --- Machine Learning in Fintech: Loan Default Prediction ---

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# --- Load the Dataset ---
data = pd.read_csv(r'C:\codebase\FinBytes\loan_data.csv')

# --- Preprocess the Data ---
X = data.drop(['default'], axis=1)  # Features
y = data['default']  # Target (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 1. Logistic Regression with Scikit-learn ---
# Train a Logistic Regression Model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Make Predictions and Evaluate the Logistic Regression Model
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = model_lr.score(X_test, y_test)
print('Logistic Regression Model Accuracy: {:.2f}%'.format(accuracy_lr * 100))

# --- 2. Gradient Boosting with XGBoost ---
# Convert data into DMatrix format required by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost model parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train the XGBoost Model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Make Predictions and Evaluate the XGBoost Model
y_pred_prob = xgb_model.predict(dtest)
y_pred_xgb = [1 if prob > 0.5 else 0 for prob in y_pred_prob]
accuracy_xgb = sum(y_pred_xgb == y_test) / len(y_test)
print('XGBoost Model Accuracy: {:.2f}%'.format(accuracy_xgb * 100))

# --- 3. Deep Learning with TensorFlow ---
# Build a Neural Network Model with Keras
model_nn = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the Neural Network Model
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Neural Network Model
history = model_nn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the Neural Network Model
test_loss, test_acc = model_nn.evaluate(X_test, y_test)
print('Neural Network Test Accuracy: {:.2f}%'.format(test_acc * 100))

# --- Visualize the Data ---

# Plot the distribution of 'income' and 'loan_amount' to understand the dataset better
plt.figure(figsize=(12, 6))

# Subplot for Income Distribution
plt.subplot(1, 2, 1)
plt.hist(data['income'], bins=10, color='skyblue', edgecolor='black')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')

# Subplot for Loan Amount Distribution
plt.subplot(1, 2, 2)
plt.hist(data['loan_amount'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()

# --- Visualize the Model Performance ---

# Plot ROC Curve for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, model_lr.predict_proba(X_test)[:,1])
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot ROC Curve for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC Curve for Neural Network
fpr_nn, tpr_nn, _ = roc_curve(y_test, model_nn.predict(X_test))
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Plot all ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.plot(fpr_xgb, tpr_xgb, color='green', lw=2, label='XGBoost (AUC = %0.2f)' % roc_auc_xgb)
plt.plot(fpr_nn, tpr_nn, color='red', lw=2, label='Neural Network (AUC = %0.2f)' % roc_auc_nn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# --- Visualize Feature Importance (XGBoost) ---
# Get and plot feature importance for the XGBoost model
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=5, title="XGBoost Feature Importance")
plt.show()

# --- Categorize Risk Levels Based on Predicted Probabilities ---
# Categorize individuals based on the predicted probabilities
risk_categories = []
for prob in y_pred_prob:
    if prob < 0.3:
        risk_categories.append('Low Risk')
    elif prob < 0.7:
        risk_categories.append('Medium Risk')
    else:
        risk_categories.append('High Risk')

# --- Ensure Correct Length of risk_categories ---
# Make sure we have the same number of rows in the risk_categories list as in the data
if len(risk_categories) != len(X_test):
    risk_categories = risk_categories[:len(X_test)]  # truncate the list if necessary

# Add the risk category to the dataframe (aligning with the test data)
data_test = X_test.copy()
data_test['risk_category'] = risk_categories

# --- Visualize the risk categories ---
plt.figure(figsize=(8, 6))
sns.countplot(x='risk_category', data=data_test, palette='viridis')
plt.title('Loan Default Risk Categories')
plt.xlabel('Risk Category')
plt.ylabel('Count')
plt.show()

# --- Identify Categories of People Likely to Default ---
# Filter and show the characteristics of people most likely to default (High Risk category)
high_risk = data_test[data_test['risk_category'] == 'High Risk']

# Display the high-risk individuals' characteristics
print("\nHigh Risk Individuals (Those Likely to Default):")
print(high_risk[['age', 'income', 'loan_amount', 'risk_category']])
