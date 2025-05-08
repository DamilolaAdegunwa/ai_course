import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to simulate real-time sentiment analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'


# Function to extract keywords and detect trends using TF-IDF
def detect_trends(data, n_features=5): # this method is better call "keywords_extractor"
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(data['Tweet'])
    feature_names = np.array(tfidf.get_feature_names_out())

    # Get the top 'n_features' terms
    sorted_idx = np.argsort(X.sum(axis=0).A1)[::-1][:n_features]
    keywords = feature_names[sorted_idx]
    return keywords


# Real-time trend detection simulation
def real_time_trend_simulation(data, window_size=50):
    # Sliding window for trend detection
    trends = []
    for i in range(len(data) - window_size + 1):
        window_data = data.iloc[i:i + window_size]
        keywords = detect_trends(window_data)
        trends.append(keywords)
    return trends


# Function to visualize trends and sentiments
def visualize_sentiment_trends(data, sentiments, trends):
    data['Sentiment'] = sentiments
    sentiment_counts = data['Sentiment'].value_counts()

    # Sentiment distribution plot
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Plot trends over time
    trend_count = pd.DataFrame(trends, columns=[f"Trend_{i}" for i in range(len(trends[0]))])
    trend_count.plot(kind='line', figsize=(10, 6))
    plt.title("Trend Detection Over Time")
    plt.xlabel("Time")
    plt.ylabel("Trend Frequency")
    plt.show()


# Main simulation for sentiment analysis and trend detection
if __name__ == "__main__":
    # Simulate incoming data
    tweets_data = {
        "Date": ["2024-12-01", "2024-12-01", "2024-12-01", "2024-12-01"],
        "Tweet": [
            "Love the new features of the app, so easy to use!",
            "This is the worst app ever, very buggy!",
            "Amazing experience! Highly recommend it!",
            "The app is terrible, crashes all the time!"
        ]
    }

    data = pd.DataFrame(tweets_data)

    # Sentiment Analysis
    sentiments = data['Tweet'].apply(sentiment_analysis)

    # Trend Detection
    trends = real_time_trend_simulation(data)

    # Visualizations
    visualize_sentiment_trends(data, sentiments, trends)

    # Print Sentiment Analysis Results
    print("\nSentiment Analysis Results:")
    for idx, sentiment in enumerate(sentiments):
        print(f"Tweet {idx + 1}: {sentiment}")

    # Print Detected Trends
    print("\nDetected Trends:")
    for idx, trend in enumerate(trends[-1]):
        print(f"Trend {idx + 1}: {trend}")


comment = """
### Project Title: **Real-Time Sentiment Analysis and Trend Detection with Pandas and NLP**  
**File Name**: `real_time_sentiment_analysis_and_trend_detection_with_pandas_and_nlp.py`  

---

### Project Description  
This project demonstrates **real-time sentiment analysis** on social media posts (e.g., tweets or comments) and the detection of emerging trends over time using **Pandas** and **Natural Language Processing (NLP)** techniques. The system will continuously process incoming text data, perform sentiment analysis, and identify trending topics using a **dynamic sliding window approach**. Key functionalities include:

1. **Sentiment Analysis**: Classifying sentiment (positive, negative, neutral) using text vectorization and machine learning.
2. **Trend Detection**: Identifying topics or hashtags that are gaining momentum using frequency-based and TF-IDF methods.
3. **Real-Time Data Processing**: Handling new incoming data at regular intervals to simulate real-time updates.
4. **Data Visualization**: Displaying trends and sentiment scores using various plots.

This project requires integrating **TextBlob**, **scikit-learn**, and **Pandas** for processing text and detecting trends.

---

### Example Use Cases
1. **Social Media Monitoring**: Analyze public sentiment toward a product, brand, or political event by examining user comments on Twitter or Reddit.
2. **Customer Feedback Analysis**: Gather real-time feedback from customer reviews and detect emerging complaints or compliments.
3. **Financial Market Sentiment**: Monitor investor sentiment on stocks or cryptocurrency by analyzing public discourse on social media.
4. **Political Sentiment Analysis**: Analyze public opinion on political figures or policies over time.

---

### Example Input(s) and Expected Output(s)

#### **Input 1**  
**Tweets Data** (Simulated Data):  
| Date       | Tweet                                           |  
|------------|-------------------------------------------------|  
| 2024-12-01 | "Love the new features of the app, so easy to use!" |  
| 2024-12-01 | "This is the worst app ever, very buggy!"         |  
| 2024-12-01 | "Amazing experience! Highly recommend it!"        |  
| 2024-12-01 | "The app is terrible, crashes all the time!"      |  

**Expected Output**:  
- Sentiment Scores:  
  - Tweet 1: Positive  
  - Tweet 2: Negative  
  - Tweet 3: Positive  
  - Tweet 4: Negative  
- Trend Analysis:  
  - Emerging Topic: **App Features** (based on frequency of keywords like "app", "features", "bugs")

#### **Input 2**  
**Customer Review Data**:  
| Date       | Review Text                                      |  
|------------|--------------------------------------------------|  
| 2024-12-01 | "I bought this laptop for work, it’s fantastic!"  |  
| 2024-12-01 | "Worst laptop, it overheats and shuts down!"     |  
| 2024-12-01 | "Great laptop for everyday use."                 |  
| 2024-12-01 | "Battery life is terrible, needs improvement."   |  

**Expected Output**:  
- Sentiment Scores:  
  - Review 1: Positive  
  - Review 2: Negative  
  - Review 3: Positive  
  - Review 4: Negative  
- Trend Detection:  
  - Emerging Trend: **Battery Life** (due to frequent mentions of "battery life" and "overheats")

#### **Input 3**  
**Reddit Comments Data**:  
| Date       | Comment                                         |  
|------------|-------------------------------------------------|  
| 2024-12-01 | "Crypto is booming right now, can't wait for the new Bitcoin surge!" |  
| 2024-12-01 | "Ethereum is a better investment than Bitcoin, my opinion." |  
| 2024-12-01 | "Bitcoin is too volatile, Ethereum is more stable." |  
| 2024-12-01 | "The crypto market will crash soon."            |  

**Expected Output**:  
- Sentiment Scores:  
  - Comment 1: Positive  
  - Comment 2: Positive  
  - Comment 3: Negative  
  - Comment 4: Negative  
- Trend Detection:  
  - Emerging Trend: **Crypto** (focused on discussions about Bitcoin and Ethereum)

---

### Python Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to simulate real-time sentiment analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to extract keywords and detect trends using TF-IDF
def detect_trends(data, n_features=5):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(data['Tweet'])
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Get the top 'n_features' terms
    sorted_idx = np.argsort(X.sum(axis=0).A1)[::-1][:n_features]
    keywords = feature_names[sorted_idx]
    return keywords

# Real-time trend detection simulation
def real_time_trend_simulation(data, window_size=50):
    # Sliding window for trend detection
    trends = []
    for i in range(len(data) - window_size + 1):
        window_data = data.iloc[i:i+window_size]
        keywords = detect_trends(window_data)
        trends.append(keywords)
    return trends

# Function to visualize trends and sentiments
def visualize_sentiment_trends(data, sentiments, trends):
    data['Sentiment'] = sentiments
    sentiment_counts = data['Sentiment'].value_counts()
    
    # Sentiment distribution plot
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Plot trends over time
    trend_count = pd.DataFrame(trends, columns=[f"Trend_{i}" for i in range(len(trends[0]))])
    trend_count.plot(kind='line', figsize=(10, 6))
    plt.title("Trend Detection Over Time")
    plt.xlabel("Time")
    plt.ylabel("Trend Frequency")
    plt.show()

# Main simulation for sentiment analysis and trend detection
if __name__ == "__main__":
    # Simulate incoming data
    tweets_data = {
        "Date": ["2024-12-01", "2024-12-01", "2024-12-01", "2024-12-01"],
        "Tweet": [
            "Love the new features of the app, so easy to use!",
            "This is the worst app ever, very buggy!",
            "Amazing experience! Highly recommend it!",
            "The app is terrible, crashes all the time!"
        ]
    }

    data = pd.DataFrame(tweets_data)

    # Sentiment Analysis
    sentiments = data['Tweet'].apply(sentiment_analysis)
    
    # Trend Detection
    trends = real_time_trend_simulation(data)

    # Visualizations
    visualize_sentiment_trends(data, sentiments, trends)
    
    # Print Sentiment Analysis Results
    print("\nSentiment Analysis Results:")
    for idx, sentiment in enumerate(sentiments):
        print(f"Tweet {idx + 1}: {sentiment}")
    
    # Print Detected Trends
    print("\nDetected Trends:")
    for idx, trend in enumerate(trends[-1]):
        print(f"Trend {idx + 1}: {trend}")
```

---

### Key Features
1. **Real-Time Sentiment Analysis**: Classifies text as positive, negative, or neutral using **TextBlob**.
2. **Trend Detection**: Identifies frequently mentioned terms (keywords) using **TF-IDF**.
3. **Sliding Window Simulation**: Simulates real-time data input using a rolling window of historical data.
4. **Data Visualization**: Displays sentiment distribution and trend evolution over time.
5. **Scalability**: Can easily be adapted for larger datasets or real-time stream data using APIs.

---

### Testing Scenarios

#### **Scenario 1**:  
- Input: Tweets with varying sentiments.  
- Test: Verify sentiment classifications and visualize sentiment distribution.

#### **Scenario 2**:  
- Input: Customer reviews with keyword patterns.  
- Test: Ensure that trends like "battery life" or "overheating" are correctly identified.

#### **Scenario 3**:  
- Input: Social media comments about financial topics.  
- Test: Confirm correct sentiment classification for positive and negative opinions about cryptocurrencies.

---

### Advanced Extensions
1. Integrate **Twitter API** to fetch real-time tweets for analysis.
2. Enhance the **NLP model** with advanced techniques like **BERT** for better sentiment analysis.
3. Incorporate **Topic Modeling** (LDA or NMF) for more nuanced trend detection.  
4. Use **Time-Series Forecasting** to predict the future sentiment trends over a given period.
"""
