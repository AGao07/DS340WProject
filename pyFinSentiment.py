from pyfin_sentiment.model import SentimentModel
import pandas as pd

model = SentimentModel("small")
df = pd.read_csv("data/processed_data/reddit_finance_posts_cleaned.csv")

A = df["content"].to_numpy(str)
contentPred = model.predict(A)
B = df["title"].to_numpy(str)
titlePred = model.predict(B)

sentiment_map = {1: "Bullish", 2: "Neutral", 3: "Bearish"}
df["contentSentiment"] = [sentiment_map[int(pred)] for pred in contentPred]
df["titleSentiment"] = [sentiment_map[int(pred)] for pred in titlePred]

df.to_csv("data/processed_data/pyFin_Predictions.csv", index=False)
print("Sentiment analysis completed and saved with descriptive labels.")
