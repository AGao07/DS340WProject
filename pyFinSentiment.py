from pyfin_sentiment.model import SentimentModel
import pandas as pd

# the model only needs to be downloaded once
#SentimentModel.download("small")

model = SentimentModel("small")
#1 is postive
#2 is neutral
#3 is negative

df = pd.read_csv("data/processed_data/reddit_finance_posts_cleaned.csv")
A=df["content"].to_numpy(str)
contentPred = model.predict(A)
contentSentiment = contentPred.tolist()
B=df["title"].to_numpy(str)
titlePred = model.predict(B)
titleSentiment = titlePred.tolist()
df["contentSentiment"]=contentSentiment
df["titleSentiment"]=titleSentiment
df.to_csv("data/processed_data/pyFin_Predictions.csv", index=False)
