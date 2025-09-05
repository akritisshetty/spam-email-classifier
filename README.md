# Spam Email Classifier

This project is a simple machine learning classifier that detects whether a message is **Spam** or **Ham (not spam)**.
It was built and tested in **Google Colab** using the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

---

## What’s Inside

* Loads and preprocesses the dataset
* Converts text into numerical features using **TF-IDF vectorization**
* Trains two models:

  * **Naive Bayes** (fast, simple baseline)
  * **Logistic Regression** (stronger, more accurate)
* Evaluates models with **accuracy, precision, recall, F1-score, and confusion matrix**
* Tests the models on **custom messages**

---

## Dataset

* **UCI SMS Spam Collection**
* \~5,500 SMS messages labeled as `ham` or `spam`

---

## Results

* **Naive Bayes Accuracy:** \~95%
* **Logistic Regression Accuracy:** \~97%
* Spammy keywords like `free`, `win`, `prize`, `congratulations` were detected strongly
* Hammy messages (like `lunch`, `project`, `meeting`) were separated well

---

## Usage

Run the notebook in **Google Colab**:

1. Upload the `.ipynb` file
2. Run all cells
3. Test with your own messages at the end

---

## Live Demo

Check out the Streamlit version of this project here:
👉 [Spam Email Classifier Live App](https://spam-email-classifier-python.streamlit.app/)

---
