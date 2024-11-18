from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from pyfin_sentiment.model import SentimentModel
import yfinance as yf

finbert_model_name = "yiyanghkust/finbert-tone"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
pyFin = SentimentModel("small")

df = pd.read_csv("data/processed_data/reddit_finance_posts_cleaned.csv")

# Negation map from parent paper
negation_map = {
        " more ": " less ", " less ": " more ", " positive ": " negative ", " increase ": " decrease ",
        " yes ": " no ", " no ": " yes ", " unable ": " able ", " able ": " unable ",
        " decrease ": " increase ", " sales ": " buy ", " sale ": " buy ", " buy ": " sale ",
        " best ": " worst ", " worst ": " best ", " larger ": " smaller ", " smaller ": " larger ",
        " large ": " small ", " small ": " large ", " good ": " bad ", " bad ": " good ",
        " high ": " low ", " low ": " high ", " down ": " up ", " up ": " down ",
        " dislike ": " like ", " like ": " dislike ", " right ": " wrong ", " wrong ": " right ",
        " a lot of ": " few ", " many ": " few ", " few ": " many ", " little ": " much ",
        " much ": " little ", " disbelieve ": " believe ", " believe ": " disbelieve ",
        " better ": " worse ", " worse ": " better ", " revenue ": " expense ", " expense ": " revenue ",
        " abandon ": " remain ", " remain ": " abandon ", " continuing ": " stopping ", " stopping ": " continuing ",
        " continue ": " stop ", " stop ": " continue ", " approve ": " refuse ", " refuse ": " approve ",
        " grew ": " decayed ", " decayed ": " grew ", " decay ": " grow ", " growth ": " decay ",
        " grow ": " decay ", " improvement ": " degeneration ", " degeneration ": " improvement ",
        " improve ": " degenerate ", " degenerate ": " improve ", " focus ": " ignore ", " ignore ": " focus ",
        " major ": " minor ", " minor ": " major ", " strong ": " weak ", " weak ": " strong ",
        " full ": " empty ", " empty ": " full ", " start ": " end ", " end ": " start ",
        " progress ": " decline ", " decline ": " progress ", " earnings ": " cost ", " cost ": " earnings ",
        " well ": " badly ", " badly ": " well ", " expect ": " dismiss ", " dismiss ": " expect ",
        " over ": " below ", " below ": " over ", " back ": " forward ", " forward ": " back ",
        " margin ": " loss ", " profit ": " loss ", " benefits ": " loss ", " income ": " loss ",
        " loss ": " profit ", " benefit ": " harm ", " harm ": " benefit ", " slightly ": " completely ",
        " completely ": " slightly ", " most ": " least ", " least ": " most ",
        " add ": " decrease ", " change ": " unchange ", " opportunities ": " changes ", " opportunity ": " change ",
        " within ": " without ", " without ": " with ", " with ": " without "
    }

def negate_text(text):
    for word, opposite in negation_map.items():
        text = text.replace(word, opposite)
    return text

def reorder_text(text):
    parts = text.split(", ")
    return ", ".join(reversed(parts)) if len(parts) > 1 else text

def concatenate_text(text1, text2):
    return text1 + " " + text2

def transitive_text(text, ticker, top_company):
    return text.replace(ticker, top_company)

def get_finbert_sentiment(text):
    sentiment_labels = {0: "Bearish", 1: "Neutral", 2: "Bullish"} # Model specific encoding
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return sentiment_labels[prediction]

def get_pyfin_sentiment(text):
    pred = pyFin.predict([text])[0]
    sentiment_map = {1: "Bullish", 2: "Neutral", 3: "Bearish"} # Model specific encoding
    return sentiment_map.get(int(pred), "Unknown")


def get_top_company(ticker):
    cleaned_ticker = ticker.strip("[]'\"")
    ticker_list = cleaned_ticker.split(", ")
    first_ticker = ticker_list[0].strip('\'')
    if first_ticker == 'RE':
        first_ticker = 'TSLA'
    stock = yf.Ticker(first_ticker)
    info = stock.info
    sector = info.get('sector')


    # Top companies by sector
    sector_top_companies = {
        'Consumer Discretionary': 'Amazon.com Inc',
        'Consumer Staples': 'Walmart',
        'Energy': 'Exxon Mobil Corp.',
        'Financials': 'JPMorgan Chase & Co.',
        'Health Care': 'Johnson & Johnson',
        'Industrials': 'Boeing Company',
        'Information Technology': 'Apple Inc.',
        'Materials': 'DowDuPont',
        'Real Estate': 'American Tower Corp A',
        'Telecommunication Services': 'AT&T Inc',
        'Utilities': 'NextEra Energy'
    }
    return sector_top_companies.get(sector, 'Tesla Inc.')


def add_consistency_checks(df):
    results = []

    for i, row in df.iterrows():
        id = row['id']
        title = row['title']
        content = row['content']
        ticker = row.get("tickers", "AAPL")
        top_company = get_top_company(ticker)

        finbert_sentiment_title = get_finbert_sentiment(title)
        pyfin_sentiment_title = get_pyfin_sentiment(title)
        finbert_sentiment_content = get_finbert_sentiment(content)
        pyfin_sentiment_content = get_pyfin_sentiment(content)

        negation_title = negate_text(title)
        reordered_title = reorder_text(title)
        combined_title = concatenate_text(title, "We expect future growth.")
        transitive_title = transitive_text(title, ticker, top_company)

        negation_content = negate_text(content)
        reordered_content = reorder_text(content)
        combined_content = concatenate_text(content, "We expect future growth.")
        transitive_content = transitive_text(content, ticker, top_company)

        results.append({
            "id": id,
            "original_title": title,
            "original_content": content,
            "ticker": ticker,
            "FinBERT_sentiment_title": finbert_sentiment_title,
            "PyFin_sentiment_title": pyfin_sentiment_title,
            "FinBERT_sentiment_content": finbert_sentiment_content,
            "PyFin_sentiment_content": pyfin_sentiment_content,
            "FinBERT_negation_title": get_finbert_sentiment(negation_title),
            "FinBERT_symmetric_title": get_finbert_sentiment(reordered_title),
            "FinBERT_additive_title": get_finbert_sentiment(combined_title),
            "FinBERT_transitive_title": get_finbert_sentiment(transitive_title),
            "FinBERT_negation_content": get_finbert_sentiment(negation_content),
            "FinBERT_symmetric_content": get_finbert_sentiment(reordered_content),
            "FinBERT_additive_content": get_finbert_sentiment(combined_content),
            "FinBERT_transitive_content": get_finbert_sentiment(transitive_content),
            "PyFin_negation_title": get_pyfin_sentiment(negation_title),
            "PyFin_symmetric_title": get_pyfin_sentiment(reordered_title),
            "PyFin_additive_title": get_pyfin_sentiment(combined_title),
            "PyFin_transitive_title": get_pyfin_sentiment(transitive_title),
            "PyFin_negation_content": get_pyfin_sentiment(negation_content),
            "PyFin_symmetric_content": get_pyfin_sentiment(reordered_content),
            "PyFin_additive_content": get_pyfin_sentiment(combined_content),
            "PyFin_transitive_content": get_pyfin_sentiment(transitive_content),
        })

    return pd.DataFrame(results)

df_consistency = add_consistency_checks(df)
df_consistency.to_csv("data/processed_data/FinBERT_PyFin_Consistency.csv", index=False)
