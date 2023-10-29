import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset (assuming you have a CSV file with 'v1' and 'v2' columns)
data = pd.read_csv(r'C:\Users\karan\Documents\Python Scripts\spam.csv', encoding='ISO-8859-1')

# Use 'v2' for the text data and 'v1' for the labels
X = data['v2']
y = data['v1']

# Vectorize the email text using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_vectorized, y)

# Function to predict if an email is spam or not
def predict_spam_or_not(email_text):
    email_vectorized = vectorizer.transform([email_text])
    prediction = classifier.predict(email_vectorized)
    return prediction[0]

# User input for an email
user_email = input("Enter an email text: ")

# Predict if it's spam or not
result = predict_spam_or_not(user_email)

if result == 'spam':
    print("This email is spam.")
else:
    print("This email is not spam.")

# Use 'v2' for the text data and 'v1' for the labels
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Vectorize the email text using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')
print(f'Classification Report:\n{report}')




