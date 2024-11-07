import praw
import pandas as pd
import re
import schedule
import time

reddit = praw.Reddit(
    client_id='VrIL6ECM-p_kqqynXK6Ucw',
    client_secret='C-ICFTty213-mlbxk-5zELH1LhtaLQ',
    user_agent='Education_NLP by /u/TakafumiKusonori'
)

processed_file = "processed_posts.txt"
try:
    with open(processed_file, "r") as file:
        processed_ids = set(line.strip() for line in file)
except FileNotFoundError:
    processed_ids = set()

subreddits = "stocks+wallstreetbets+investing"
subreddit = reddit.subreddit(subreddits)

ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

def scrape_and_process_posts():
    new_posts = []

    for post in subreddit.top(limit=500):
        if post.id not in processed_ids:
            tickers_in_title = ticker_pattern.findall(post.title)
            tickers_in_content = ticker_pattern.findall(post.selftext)
            tickers = list(set(tickers_in_title + tickers_in_content))
            post_content_cleaned = post.selftext.replace("\n", " ")

            new_posts.append({
                "id": post.id,
                "title": post.title,
                "score": post.score,
                "content": post_content_cleaned,
                "url": post.url,
                "subreddit": post.subreddit.display_name,
                "tickers": ", ".join(tickers)
            })

            processed_ids.add(post.id)

    with open(processed_file, "a") as file:
        for post in new_posts:
            file.write(post["id"] + "\n")

    if new_posts:
        df = pd.DataFrame(new_posts)
        print("New posts processed. Saving to CSV...")
        print(df)
        df.to_csv("data/raw_data/reddit_finance_posts.csv", mode='a', header=False, index=False)
    else:
        print("No new posts found.")
schedule.every(1).hours.do(scrape_and_process_posts)

while True:
    schedule.run_pending()
    time.sleep(60)
