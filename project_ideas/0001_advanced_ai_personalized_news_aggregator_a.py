import os
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
from apikey import newsapi_apikey
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
# Initialize Flask app
app = Flask(__name__)

# Constants
NEWS_API_KEY = newsapi_apikey  # Replace with your NewsAPI key
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'
# STOPWORDS = set(stopwords.words('english'))


# Functions
def fetch_news(keywords, language='en'):
    """Fetch news articles from NewsAPI."""
    params = {'q': keywords, 'language': language, 'apiKey': NEWS_API_KEY}
    response = requests.get(NEWS_API_URL, params=params)
    print(f"here is the response.json() frm fetch_news: {response.json()}")
    if response.status_code == 200:
        return response.json().get('articles', [])
    return []


def clean_text(text):
    """Clean and preprocess text."""
    tokens = word_tokenize(text.lower())
    # return ' '.join([word for word in tokens if word.isalnum() and word not in STOPWORDS])
    return ' '.join([word for word in tokens if word.isalnum()])


def extract_topics(articles, num_topics=5, num_words=10):
    """Perform topic modeling using NMF."""
    texts = [clean_text(article['content'] or '') for article in articles if article['content']]
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=num_topics, random_state=42)
    # w = nmf.fit_transform(tfidf_matrix)
    h = nmf.components_
    topics = [
        [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:][::-1]]
        for topic in h
    ]
    return topics


def recommend_articles(user_profile, articles):
    """Recommend articles based on user profile using cosine similarity."""
    articles_text = [clean_text(article['content'] or '') for article in articles if article['content']]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(articles_text + [user_profile])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    recommended_indices = similarity_scores.argsort()[0][-5:][::-1]
    return [articles[i] for i in recommended_indices]


def analyze_sentiment_old(articles):
    """Perform sentiment analysis on articles."""
    sia = SentimentIntensityAnalyzer()
    print(f" the article object passed to the analyze_sentiment method is {articles}")
    for article in articles:
        sentiment = sia.polarity_scores(article.get('description', ''))
        article['sentiment'] = sentiment
    return articles


def analyze_sentiment(articles):
    # Ensure the VADER lexicon is downloaded
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    print(f" the article object passed to the analyze_sentiment method is {articles}")
    for article in articles:
        sentiment = sia.polarity_scores(article["content"])
        article["sentiment"] = sentiment["compound"]
    return articles


@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for recommending articles."""
    user_data = request.json
    keywords = user_data.get('interests', '')
    print(f"and the keyword is: {keywords}")
    sentiment_filter = user_data.get('preferred_sentiment', 'positive')
    articles = fetch_news(keywords)
    articles = analyze_sentiment(articles)
    print(f"and the articles value in the /recommend endpoint is: {articles}")
    # articles = [article for article in articles if article['sentiment']['compound'] > 0.5]
    articles = [article for article in articles]
    recommended_articles = recommend_articles(keywords, articles)
    return jsonify({'recommendations': recommended_articles})


if __name__ == "__main__":
    app.run(debug=True)
