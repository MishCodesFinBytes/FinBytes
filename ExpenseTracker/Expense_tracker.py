'''
Variation 1: Basic Expense Tracker

'''

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the expenses data from a CSV file
df = pd.read_csv(r"C:\codebase\FinBytes\expenses.csv")

# Create a pie chart to visualize the breakdown of expenses by category
plt.figure(figsize=(8, 8))
plt.pie(
    df['Amount'], 
    labels=df['Category'], 
    autopct='%1.1f%%', 
    startangle=140
)
plt.title("Monthly Expenses Breakdown")
plt.show()


'''
Variation 2: Advanced Expense Tracker with Predictive Analysis
'''

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Read the expenses data from a CSV file
df = pd.read_csv(r"C:\codebase\FinBytes\expenses.csv")

# Ensure Month is numeric (e.g., 1 for January, 2 for February, etc.)
df['Month'] = pd.to_datetime(df['Month'], format='%B').dt.month

# Split the data into training and testing sets
X = df['Month'].values.reshape(-1, 1)
y = df['Amount'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future expenses for hypothetical months
future_months = np.array([5, 6, 7, 8, 9]).reshape(-1, 1)  # May to September
future_expenses = model.predict(future_months)

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Amount'], label="Actual Expenses", marker='o')
plt.plot(future_months, future_expenses, label="Predicted Expenses", linestyle='--', marker='x')
plt.xlabel("Month")
plt.ylabel("Expense Amount")
plt.title("Expense Prediction")
plt.legend()
plt.grid()
plt.show()


'''
Variation 3: Expense Categorization with NLP
'''

# Import necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure you have downloaded the NLTK data files
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Read the expenses data from a CSV file
df = pd.read_csv(r"C:\codebase\FinBytes\expenses.csv")

# Function to preprocess text descriptions
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into a sentence
    return ' '.join(filtered_tokens)

# Apply the preprocessing function to the expense descriptions
df['Processed_Description'] = df['Description'].apply(preprocess)

# Define broader keywords for each category
category_keywords = {
    'Food': ['food', 'lunch', 'dinner', 'cafe', 'restaurant', 'meal'],
    'Groceries': ['groceries', 'supermarket', 'shopping'],
    'Transportation': ['transportation', 'bus', 'fuel', 'car', 'travel', 'commute'],
    'Entertainment': ['entertainment', 'movie', 'concert', 'tickets', 'theater']
}

# Categorize expenses based on the keywords
def assign_category(description):
    for category, keywords in category_keywords.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'  # Default category

df['Category'] = df['Processed_Description'].apply(assign_category)

# Create a pie chart for categorized expenses
plt.figure(figsize=(8, 8))
plt.pie(
    df.groupby('Category')['Amount'].sum(), 
    labels=df.groupby('Category')['Amount'].sum().index, 
    autopct='%1.1f%%', 
    startangle=140
)
plt.title("Categorized Monthly Expenses")
plt.show()
