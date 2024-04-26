import streamlit as st
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import string

nltk.download('punkt')


@st.cache_data
def load_model():
    with open('sentiment_classifier.pkl', "rb") as f:
        model=pickle.load(f)
    return model


def preprocess_tweet(tweet):
    # Tokenize the tweet
    tokens = word_tokenize(tweet)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if
              (word.lower() not in stop_words and word.lower() not in string.punctuation)]

    return tokens


def predict_sentiment(tweet, classifier):
    # Preprocess the tweet
    preprocessed_tweet = preprocess_tweet(tweet)

    # Classify the tweet using the classifier
    sentiment = classifier.classify(dict([token, True] for token in preprocessed_tweet))

    return sentiment

def main():
    st.title('Twitter Sentiment Analysis')
    tweet=st.text_input("Enter your tweet to generate it's sentiment")
    if st.button("Generate"):
        if tweet:
            classifier = load_model()
            sentiment = predict_sentiment(tweet, classifier)
            st.write("Prediction: ", sentiment)
        else:
            st.warning("Please enter a tweet.")


if __name__ == "__main__":
    main()
