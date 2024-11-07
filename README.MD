Parent Papers

FinBERT.

FinTRUST.

PyFin-Sentiment.

Analysis on social media text analysis.

pip install -r requirements.txt

Scrape.py gets 100 posts (Rate limit by 100 posts for each run and limited to 60 runs per hour) from r/investing, r/stocks, and r/wallstreetbets.

Each run can either be sorted by "relevance", "hot", "top", "new", or "comments"

There is a running count of downloaded post ids in processed_posts.txt, and it is a large aggregate of posts.

Running Clean.py will only involve posts that specifically mention a stock, and make clean the formatting.

After running both functions, there will be "reddit_finance_posts.csv" in data/raw_data and "reddit_finance_posts_cleaned.csv" in data/processed_data.