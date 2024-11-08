from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

df = pd.read_csv("data/processed_data/reddit_finance_posts_cleaned.csv")

sentiment_labels = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

def batch_sentiment_analysis(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # No gradients needed for inference
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        sentiments = [sentiment_labels[pred.item()] for pred in predictions]
    return sentiments

df["contentSentiment"] = batch_sentiment_analysis(df["content"].tolist())
df["titleSentiment"] = batch_sentiment_analysis(df["title"].tolist())

df.to_csv("data/processed_data/FinBERT_Predictions.csv", index=False)
print("Sentiment analysis completed and saved.")
