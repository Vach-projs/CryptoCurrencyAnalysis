This project is a real-time cryptocurrency analysis and forecasting system that combines time series models, volatility analysis, and NLP-based sentiment tracking to help users make smarter trading and portfolio decisions. Built using Python and Streamlit, it fetches data from Binance, Yahoo Finance, and CoinGecko, performs advanced financial modeling (Prophet, GARCH), and visualizes everything on a sleek GUI. It also scrapes social media/news sentiment using BERT-based models, correlates it with market trends, and provides a holistic view of price movements, risk, and trading signals—all in one place.

The datacollect.py file collects data from the Yahoo finance API and CoinGecko API collects the realtime price.

The preprocess.py contains some preprocessing steps and feature engineering.

eda.py contains the EDA done on the data.

volatility.py contains code for the volatility analysis with two models, GARCH and EGARCH.

sentiment.py contains the sentiment analysis done on information collected using Reddit API and News.org API. An attempt was made to get information from X API but it didn't really work. 

time_series.py contains the model training using Prophet model. The data was based on all the previous features, volatility, sentiment analysis and feature engineering along with the historical data. The data was extracted into a CSV file after every step for the time series model. 

The results were showcased using Streamlit. 
