# Import libraries for machine learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Example dataset: financial text data with sentiment labels (1: Positive, 0: Negative)
text_data = [
    "Stock prices surged after the Federal Reserve's announcement.",
    "The market is experiencing significant volatility.",
    "Company earnings exceeded analyst expectations.",
    "Investors are worried about the rising inflation rates.",
    "The economic outlook remains uncertain."
]
labels = [1, 0, 1, 0, 0]  # Corresponding sentiment labels

# Step 1: Vectorize the text data into numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_data)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 3: Train a Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Step 4: Predict sentiment on test data
predictions = nb.predict(X_test)

# Print predictions and corresponding true labels
print("\nSentiment Analysis using Machine Learning:")
for i, prediction in enumerate(predictions):
    print(f"Text: {text_data[i]}")
    print(f"Predicted Sentiment: {'Positive' if prediction == 1 else 'Negative'}, Actual Sentiment: {'Positive' if y_test[i] == 1 else 'Negative'}\n")
