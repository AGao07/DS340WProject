import praw
import pandas as pd
import re
import getTickers

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

new_posts = []

for post in subreddit.new(limit=500):  # Fetch up to 100 posts and
    if post.id not in processed_ids:
        # Extract tickers from title and content
        tickers_in_title = ticker_pattern.findall(post.title)
        tickers_in_content = ticker_pattern.findall(post.selftext)
        tickers = list(set(tickers_in_title + tickers_in_content))  # Combine and remove duplicates
        post_content_cleaned = post.selftext.replace("\n", " ")

        new_posts.append({
            "id": post.id,
            "title": post.title,
            "score": post.score,
            "content": post.selftext,
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
    print("DataFrame created successfully:")
    print(df)
    df.to_csv("data/raw_data/reddit_finance_posts.csv", mode='a', header=False, index=False)
else:
    print("No new posts to save.")