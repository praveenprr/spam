import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv('/content/spam (3).csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Assuming 'v1' is label and 'v2' is text
df.columns = ['label', 'text']
data = {
    'text': [
        'Free money now!!!',
        'Hi, how are you?',
        'Claim your free prize',
        'Meeting tomorrow at 10am',
        'You have won a free gift card',
        'Can we reschedule our meeting?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)
X = df['text']  # Features (Email text)
y = df['label']  # Labels (Spam or Ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


import matplotlib.pyplot as plt # import the necessary module
import pandas as pd

# Calculate the counts of 'spam' and 'ham' labels
label_counts = df['label'].value_counts()

plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Spam vs Ham Email Distribution')
plt.xlabel('Label')  # Changed from 'recall' for clarity
plt.ylabel('Count')  # Changed from 'support' for clarity
plt.xticks(rotation=0)
plt.show()
