# === Install dependencies ===
!pip install praw vaderSentiment --quiet

#!pip install praw vaderSentiment --quiet

import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# === VADER Setup ===
analyzer = SentimentIntensityAnalyzer()

# === Reddit API Setup ===
reddit = praw.Reddit( #use async praw if you are going to make a lot of requests in a small amount of time. If you are going to use praw as is, then give it at least ten to fifteen minutes before making a request or it will scraping info for you
    client_id='YOUR_CLIENT_ID', #you can create your own app on Reddit for scraping info, just look it up its easy
    client_secret='YOUT_CLIENT_SECRET',
    user_agent='USER_AGENT' #this is optional but recommended to be named 
)

# === Date Range Setup ===
start_date = datetime(2025, 1, 1)
end_date = datetime.utcnow()
date_range = pd.date_range(start=start_date, end=end_date)

# === Placeholder for daily sentiment scores ===
daily_sentiment = {d.date(): [] for d in date_range}

# === Subreddits to scan ===
subreddits = ['CryptoCurrency', 'Bitcoin', 'CryptoMarkets']

print("Scraping Reddit...")

for sub in subreddits:
    print(f"Searching r/{sub}")
    try:
        for submission in reddit.subreddit(sub).search('bitcoin', sort='new', time_filter='all', limit=1000):
            created = datetime.utcfromtimestamp(submission.created_utc).date()
            if created in daily_sentiment:
                text = submission.title + " " + submission.selftext
                score = analyzer.polarity_scores(text)['compound']
                daily_sentiment[created].append(score)

    except Exception as e:
        print(f"Error in r/{sub}: {e}")

# === Average Sentiment Per Day ===  #becuase there were too many values for a single day. doesn't really change the sentiment drastically, the sentiment is mostly positive most of the time anyway, rarely found many negative sentiments 
sentiment_data = []
for date, scores in daily_sentiment.items():
    avg_score = sum(scores) / len(scores) if scores else None
    sentiment_data.append({'date': date, 'reddit_sentiment': avg_score})

# === Save to CSV ===
sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df.to_csv('/content/reddit_sentiment.csv', index=False)
print("Done! Saved to reddit_sentiment.csv")

import requests
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Set your NewsAPI key here
NEWS_API_KEY = "YOUR_KEY" #same thing here its very easy, just log onto news.org

# Initialize VADER
vader = SentimentIntensityAnalyzer()

def get_articles_for_date(date, keyword="bitcoin"):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "from": date,
        "to": date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200 or "articles" not in data:
        print(f"Error on {date}: {data.get('message', 'Unknown error')}")
        return []

    return data["articles"]

def extract_sentiment_from_articles(articles):
    sentiments = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        sentiment = vader.polarity_scores(text)['compound']
        sentiments.append(sentiment)
    return sentiments

def scrape_news_sentiment(keyword="bitcoin", start_date="2025-03-21", end_date="2025-04-21"):
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    all_data = []

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Processing {date_str}")
        try:
            articles = get_articles_for_date(date_str, keyword)
            sentiments = extract_sentiment_from_articles(articles)
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
            all_data.append({"date": date_str, "news_sentiment": avg_sentiment})
            time.sleep(1.2)  #Be gentle with NewsAPI's rate limit
        except Exception as e:
            print(f"Error on {date_str}: {e}")
            all_data.append({"date": date_str, "news_sentiment": None})
        current_date += timedelta(days=1)

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    print("News sentiment scraping complete!")
    return df

# Run the scraper
news_sentiment_df = scrape_news_sentiment()
news_sentiment_df.to_csv("news_sentiment.csv", index=False)

import pandas as pd

# Load your two sentiment CSVs
reddit_df = pd.read_csv('/content/reddit_sentiment.csv', parse_dates=['date'])
news_df = pd.read_csv('/content/news_sentiment.csv', parse_dates=['date'])

# Create a date range from Jan 1 to Apr 21
date_range = pd.date_range(start='2025-01-01', end='2025-04-21')
merged_df = pd.DataFrame({'date': date_range})

# Merge both sentiment sources
merged_df = merged_df.merge(reddit_df, on='date', how='left')
merged_df = merged_df.merge(news_df, on='date', how='left')

# Calculate final sentiment score: average if both exist, else keep the one that exists
merged_df['sentiment_score'] = merged_df[['reddit_sentiment', 'news_sentiment']].mean(axis=1)

# Optional: drop the individual columns if you want just the final sentiment
# merged_df = merged_df[['date', 'sentiment_score']]

# Save the final merged sentiment CSV
merged_df.to_csv('/content/merged_sentiment.csv', index=False)
print("Merged sentiment CSV saved at: /content/merged_sentiment.csv")

import pandas as pd

# Load your base feature CSV and the merged sentiment CSV
btc_df = pd.read_csv('/content/btc_vol.csv', parse_dates=['date'])
sentiment_df = pd.read_csv('/content/merged_sentiment.csv', parse_dates=['date'])

# Merge only the sentiment_score column on 'date'
btc_df = btc_df.merge(sentiment_df[['date', 'sentiment_score']], on='date', how='left')

# Save the final merged file
btc_df.to_csv('/content/btc_sentimentn.csv', index=False)
print("Final dataset saved at: /content/btc_sentiment.csv")
