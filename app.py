import streamlit as st
import pandas as pd
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Download dataset if not present
# ----------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, "smsspamcollection.zip")
    import zipfile
    with zipfile.ZipFile("smsspamcollection.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# ----------------------------
# Load and train model
# ----------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE, sep='\t', names=['label', 'message'])
    
    X = data['message']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

model, vectorizer = train_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📧 Spam Email Classifier")
st.write("Paste a message below and I’ll predict whether it’s spam or not.")

user_input = st.text_area("Enter a message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please type a message first.")
    else:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        
        if prediction == "spam":
            st.error("🚨 This looks like SPAM!")
        else:
            st.success("✅ This looks like HAM (not spam).")
