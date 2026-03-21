"""
Clustering in Python for Finance
Sections:
1. Customer Segmentation
2. Fraud Detection
3. Risk Assessment
"""

# --- Import required libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# --- 1. Customer Segmentation ---
# Load dataset
df_customers = pd.read_csv(r'C:\codebase\FinBytes\customer_transactions.csv')

# Select relevant features
features = ['age', 'income', 'transaction_amount']
X_customers = df_customers[features]

# Scale the features
scaler = StandardScaler()
X_customers_scaled = scaler.fit_transform(X_customers)

# Apply K-Means Clustering
kmeans_customers = KMeans(n_clusters=3, random_state=42)
df_customers['Cluster'] = kmeans_customers.fit_predict(X_customers_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = df_customers[df_customers['Cluster'] == cluster]
    plt.scatter(
        cluster_data['income'], 
        cluster_data['transaction_amount'], 
        label=f'Cluster {cluster}'
    )
plt.title("Customer Segmentation Clusters", fontsize=14)
plt.xlabel("Income", fontsize=12)
plt.ylabel("Transaction Amount", fontsize=12)
plt.legend()
plt.grid(alpha=0.5)
plt.show()

# --- 2. Fraud Detection ---
# Load dataset
df_fraud = pd.read_csv(r'C:\codebase\FinBytes\credit_card_transactions.csv')

# Select relevant features
features_fraud = ['time', 'amount']
X_fraud = df_fraud[features_fraud]

# Scale the features
X_fraud_scaled = scaler.fit_transform(X_fraud)

# Apply K-Means Clustering
kmeans_fraud = KMeans(n_clusters=2, random_state=42)
df_fraud['Label'] = kmeans_fraud.fit_predict(X_fraud_scaled)

# Visualize transactions labeled as normal or fraudulent
plt.figure(figsize=(10, 6))
for label in range(2):
    label_data = df_fraud[df_fraud['Label'] == label]
    plt.scatter(
        label_data['time'], 
        label_data['amount'], 
        label=f'{"Fraud" if label else "Normal"}'
    )
plt.title("Fraud Detection Clusters", fontsize=14)
plt.xlabel("Transaction Time", fontsize=12)
plt.ylabel("Transaction Amount", fontsize=12)
plt.legend()
plt.grid(alpha=0.5)
plt.show()

# --- 3. Risk Assessment ---
# Load dataset
df_risk = pd.read_csv(r'C:\codebase\FinBytes\credit_risk.csv')

# Select relevant features
features_risk = ['credit_score', 'income', 'debt']
X_risk = df_risk[features_risk]

# Perform Hierarchical Clustering
Z = linkage(X_risk, method='ward')

# Plot the dendrogram 
plt.figure(figsize=(14, 8))  # Increased figure size for better readability
dendrogram(
    Z,
    labels=df_risk.index,
    leaf_rotation=90,  # Rotate labels for better readability
    leaf_font_size=10  # Adjust font size
)
plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
plt.xlabel("Index", fontsize=1)
plt.ylabel("Distance", fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()  # Automatically adjust subplot parameters for better fit
plt.show()


# Adjust the dendrogram to make labels readable
plt.figure(figsize=(20, 10))  # Wider figure for more room
dendrogram(
    Z,
    labels=df_risk.index,
    leaf_rotation=90,  # Keep 90-degree rotation
    leaf_font_size=8,  # Small font size to fit labels
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=30  # Adjust this number to limit the number of leaves shown
)
plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
plt.xlabel("Clustered Data Points", fontsize=14)
plt.ylabel("Distance", fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()  # Ensure everything fits well
plt.show()
