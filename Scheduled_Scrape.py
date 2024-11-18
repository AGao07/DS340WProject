import praw
import pandas as pd
import schedule
import time
import json
import logging
from pathlib import Path
import getTickers
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

reddit = praw.Reddit(
    client_id=os.getenv("PRAW_client_id"),
    client_secret=os.getenv("PRAW_client_secret"),
    user_agent=os.getenv("PRAW_user_agent")
)

processed_file = Path("processed_posts.txt")
processed_ids = set()

if processed_file.exists():
    with open(processed_file, "r") as file:
        processed_ids = set(line.strip() for line in file)

subreddits = "stocks+wallstreetbets+investing"
subreddit = reddit.subreddit(subreddits)
valid_tickers = set(getTickers.get_sp500_tickers())  # Fetch valid tickers


def extract_valid_tickers(text, valid_tickers):
    words = text.split()
    tickers = [word for word in words if word in valid_tickers]
    return tickers


def scrape_and_process_posts():
    new_posts = []

    try:
        for post in subreddit.hot(limit=6000):
            if post.id in processed_ids:
                continue

            tickers = list(set(extract_valid_tickers(post.title, valid_tickers) +
                               extract_valid_tickers(post.selftext, valid_tickers)))
            if not tickers:
                continue

            new_posts.append({
                "id": post.id,
                "title": post.title,
                "score": post.score,
                "content": post.selftext.replace("\n", " "),
                "url": post.url,
                "subreddit": post.subreddit.display_name,
                "tickers": ", ".join(tickers)
            })

            processed_ids.add(post.id)

        with open(processed_file, "w") as file:
            json.dump(list(processed_ids), file)

        if new_posts:
            df = pd.DataFrame(new_posts)
            df.to_csv("data/raw_data/reddit_finance_posts.csv", mode='a',
                      header=not Path("data/raw_data/reddit_finance_posts.csv").exists(), index=False)
            logging.info(f"{len(new_posts)} new posts processed and saved to CSV.")
        else:
            logging.info("No new posts found.")

    except Exception as e:
        logging.error(f"Error during scraping: {e}")


schedule.every(1).hours.do(scrape_and_process_posts)

while True:
    schedule.run_pending()
    time.sleep(60)
