import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK data is downloaded
nltk.download('punkt_tab')
# nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
ps = PorterStemmer()

def transform_message(message):
    message = message.lower()
    message = word_tokenize(message)  # Use the correct function

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf = TfidfVectorizer(max_features=1)
# Train your vectorizer on your dataset and fit it
# tfidf.fit(your_training_data)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
# Load model and vectorizer
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('Spam_message.pkl', 'rb'))

# Streamlit app
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_message(input_sms)
    vector_input = tfidf.fit_transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
