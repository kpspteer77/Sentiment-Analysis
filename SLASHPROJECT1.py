# Install necessary packages
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
except ImportError:
    install('pandas')
    install('scikit-learn')
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report

# Step 1: Load and Preprocess Data

# Sample data for demonstration (replace this with your actual dataset)
data = pd.DataFrame({
    'review': [
        'The food was absolutely wonderful, from preparation to presentation, very pleasing.',
        'I did not like the food at all.',
        'Average service and food.',
        'The ambiance was wonderful, but the food was below average.',
        'Fantastic experience! Will come again.',
        'Not worth the price.',
        'Totally loved the desserts.',
        'The place was dirty and the service was slow.',
        'Great place for a family dinner.',
        'The food was just okay.'
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'neutral', 'positive',
        'negative', 'positive', 'negative', 'positive', 'neutral'
    ]
})

# Preprocess data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Additional preprocessing steps can be added here (e.g., removing punctuation, stopwords, etc.)
    return text

data['review'] = data['review'].apply(preprocess_text)

# Step 2: Build and Train the Model

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Create a pipeline for vectorization and classification
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Step 3: Prediction Function
def predict_sentiment(review):
    # Preprocess the review
    review = preprocess_text(review)
    # Predict sentiment
    return model.predict([review])[0]

# Step 4: Test the Model
# Test reviews
test_reviews = [
    "The food was absolutely wonderful, from preparation to presentation, very pleasing.",
    "I did not like the food at all.",
    "Average service and food."
]

for review in test_reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")

# Step 5: Evaluate Model Performance (Optional)
# Evaluate the model using the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
